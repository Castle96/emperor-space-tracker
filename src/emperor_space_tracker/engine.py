"""One polling pass, and the supervision loop that repeats it.

The split is deliberate. :class:`PollEngine.run_once` is a pure-ish function of
``(config, store, now)`` that performs exactly one pass, returns a
:class:`PollResult` and never raises for a recoverable condition. That makes the
whole data path testable without signals, without sleeping and without systemd.

:class:`Daemon` owns only what a supervisor needs: the sleep between passes,
signal handling, the failure counter, the heartbeat and the exit code. It knows
nothing about NOAA, SAR or Discord.

Exit codes are a contract with systemd, so they are explicit:

==== ==========================================================================
Code Meaning
==== ==========================================================================
0    Clean shutdown on SIGTERM, or the loop was never started.
1    Unrecoverable: configuration invalid, state directory unwritable, or the
     failure counter exceeded ``consecutive_failures_before_page``.
2    Interrupted by a signal that is not a shutdown request.
==== ==========================================================================
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import FrameType
from typing import Any

from .alerts import (
    Alert,
    AlertDispatcher,
    NullNotifier,
    RuleContext,
    RuleEngine,
    build_default_rules,
)
from .alerts.notifier import build_dispatcher
from .config import Config
from .errors import StoreError, TrackerError
from .memory import collect, current_rss_bytes
from .models import (
    Colony,
    SarScene,
    Severity,
    SourceHealth,
    SpaceWeatherSnapshot,
    utc_now,
)
from .net import HttpClient
from .sources import biological, sar, space_weather
from .store import Store, open_default_store

__all__ = [
    "Daemon",
    "PollEngine",
    "PollResult",
]

_LOG = logging.getLogger("emperor.engine")


@dataclass(slots=True)
class PollResult:
    """Outcome of a single polling pass.

    Attributes
    ----------
    started_at
        When the pass began.
    finished_at
        When the pass completed.
    snapshot
        The space weather snapshot, or ``None`` if the section was disabled or
        the fetch failed outright.
    scenes
        SAR scenes produced.
    colonies
        Colonies within range.
    health
        Per-source health entries.
    alerts
        Alerts raised.
    degraded
        Names of the components that failed.
    ok
        ``True`` if at least the primary data paths succeeded.
    duration_ms
        Wall-clock duration.
    rss_after_bytes
        RSS after the pass and the memory reclaim, for trend reporting.
    """

    started_at: datetime
    finished_at: datetime
    snapshot: SpaceWeatherSnapshot | None
    scenes: list[SarScene] = field(default_factory=list)
    colonies: list[Colony] = field(default_factory=list)
    health: list[SourceHealth] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    degraded: tuple[str, ...] = ()
    ok: bool = True
    duration_ms: int = 0
    rss_after_bytes: int | None = None

    def summary(self) -> str:
        """Return a one-line summary suitable for a log line or a status line.

        Returns
        -------
        str
        """
        weather = self.snapshot.headline() if self.snapshot else "space weather disabled"
        ice = "none" if not self.scenes else f"{len(self.scenes)} scene(s)"
        worst = ""
        if self.scenes:
            staged = sorted(self.scenes, key=lambda s: s.open_water_fraction, reverse=True)
            worst = f", worst {staged[0].classify()}"
        if self.ok and not self.degraded:
            mark = "ok"
        else:
            mark = f"degraded:{','.join(self.degraded) or 'unknown'}"
        return (
            f"[{mark}] {weather} | {ice}{worst} | "
            f"{len(self.alerts)} alert(s) | {self.duration_ms} ms"
        )


class PollEngine:
    """Executes one polling pass across every enabled source.

    Parameters
    ----------
    config
        The loaded configuration.
    store
        Destination store.
    http
        Optional pre-built HTTP client. One is created if omitted.
    dispatcher
        Optional alert channel. A :class:`~emperor_space_tracker.alerts.NullNotifier`
        is used if omitted.
    rules
        Optional custom rule set, for tests.

    Examples
    --------
    >>> from emperor_space_tracker.config import load_config
    >>> from emperor_space_tracker.store import Store
    >>> cfg = load_config(use_user_config=False)
    >>> engine = PollEngine(cfg, Store(":memory:"))
    >>> result = engine.run_once(dry_run=True)  # doctest: +SKIP
    >>> result.ok  # doctest: +SKIP
    True
    """

    def __init__(
        self,
        config: Config,
        store: Store,
        *,
        http: HttpClient | None = None,
        dispatcher: AlertDispatcher | None = None,
        rules: list[Any] | None = None,
    ) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.config = config
        self.store = store
        self.http = http or HttpClient(
            timeout=config.daemon.http_timeout_seconds,
            max_retries=config.alerts.max_retries,
        )
        self.dispatcher: AlertDispatcher = dispatcher or NullNotifier()
        # Both source clients are built once and reused for the life of the
        # engine. `ColonyCensusClient` memoises the parsed census catalogue and
        # the resolved GBIF taxon key, so constructing it per pass threw both
        # caches away and re-read and re-parsed `data/colonies.toml` on every
        # single poll. The config is fixed for the life of a daemon, so there
        # is nothing for a per-pass rebuild to pick up.
        self._space_weather = space_weather.SpaceWeatherClient(
            self.http,
            track_f107=self.config.space_weather.track_f107,
        )
        self._census = biological.ColonyCensusClient(
            self.http,
            species=self.config.colonies.species,
            max_distance_km=self.config.colonies.max_distance_km,
            max_colonies=self.config.colonies.max_colonies,
        )

        self.rule_engine = RuleEngine(
            rules
            if rules is not None
            else build_default_rules(
                config.space_weather,
                cooldown_seconds=config.alerts.cooldown_seconds,
                sar=config.sar,
            ),
            store=store,
            min_severity=config.alerts.severity_floor(),
        )
        self._poll_count = 0
        self._last_colony_refresh = 0
        self._known_colonies: list[Colony] = []

    @property
    def poll_count(self) -> int:
        """Return how many passes this engine has run."""
        return self._poll_count

    def run_once(self, *, dry_run: bool = False, now: datetime | None = None) -> PollResult:
        """Perform one complete polling pass.

        Parameters
        ----------
        dry_run
            Evaluate and log every alert but do not transmit, and do not write
            to the store. Backs ``est poll --dry-run`` and is the right way to
            validate a threshold change against live data before trusting it.
        now
            Injectable clock for deterministic tests.

        Returns
        -------
        PollResult
            The pass outcome. Recovers from every :class:`TrackerError`; an
            unexpected exception propagates, because that is a bug and letting
            systemd restart the process is the correct response.
        """
        started = now or utc_now()
        wall = time.monotonic()
        health: list[SourceHealth] = []
        degraded: list[str] = []
        snapshot: SpaceWeatherSnapshot | None = None
        scenes: list[SarScene] = []
        colonies: list[Colony] = []

        # -- space weather --------------------------------------------------
        if self.config.space_weather.enabled:
            client = self._space_weather
            try:
                snapshot, sw_health = client.poll()
                health.extend(sw_health)
                degraded.extend(snapshot.degraded)
                stale = client.is_stale(snapshot, self.config.space_weather.max_sample_age_seconds)
                if stale:
                    degraded.append("stale")
            except TrackerError as exc:
                degraded.append("space_weather")
                health.append(
                    SourceHealth("swpc", False, 0, f"poll failed: {exc}", 0, started)
                )
                _LOG.error("space weather poll failed: %s", exc)

        # -- colonies -------------------------------------------------------
        census = self._census
        due = (
            self._poll_count % max(1, self.config.colonies.refresh_every_polls) == 0
            or not self._known_colonies
        )
        try:
            if due:
                colonies, bio_health = census.poll(
                    latitude=self.config.site.latitude,
                    longitude=self.config.site.longitude,
                    include_gbif=self.config.colonies.include_gbif,
                )
                health.extend(bio_health)
                if any(not h.ok for h in bio_health):
                    degraded.append("colonies")
                self._known_colonies = colonies
                self._last_colony_refresh = self._poll_count
            else:
                colonies = self._known_colonies
                health.append(
                    SourceHealth(
                        "scar.catalogue", True, 0,
                        f"using cached census from pass {self._last_colony_refresh}",
                        len(colonies), started,
                    )
                )
        except TrackerError as exc:
            degraded.append("colonies")
            colonies = self._known_colonies
            health.append(SourceHealth("colonies", False, 0, str(exc), 0, started))

        # -- SAR ------------------------------------------------------------
        if self.config.sar.enabled and colonies:
            sar_client = sar.SarClient(self.http, self.config.sar)
            try:
                scenes, sar_health = sar_client.poll(
                    colonies=colonies,
                    scenes_per_colony=1,
                    latest_acquisition=self.store.latest_acquisitions() if self.store else None,
                )
                health.extend(sar_health)
                if any(h.status == "down" for h in sar_health):
                    degraded.append("sar")
            except TrackerError as exc:
                degraded.append("sar")
                health.append(SourceHealth("sar", False, 0, str(exc), 0, started))
                _LOG.error("SAR poll failed: %s", exc)

        # -- persist --------------------------------------------------------
        if not dry_run:
            self._persist(snapshot, scenes, colonies, health)

        # -- rules ----------------------------------------------------------
        stale_products: tuple[str, ...] = ()
        if snapshot is not None and "stale" in degraded:
            stale_products = ("one or more space weather products",)

        context = RuleContext(
            snapshot=snapshot or SpaceWeatherSnapshot(observed_at=started),
            scenes=tuple(scenes),
            colonies=tuple(colonies),
            now=started,
            stale_products=stale_products,
        )
        outcomes = self.rule_engine.evaluate(context)

        raised: list[Alert] = []
        for outcome in outcomes:
            if outcome.alert is None:
                if outcome.state in {"active", "pending", "suppressed"}:
                    _LOG.info("rule %-22s %-10s %s", outcome.rule_id, outcome.state, outcome.detail)
                continue
            if outcome.state == "cleared":
                _LOG.info("rule %-22s CLEARED", outcome.rule_id)
            else:
                _LOG.warning("rule %-22s FIRED  %s", outcome.rule_id, outcome.alert.title)
            if not dry_run:
                self._deliver(outcome.rule_id, outcome.alert)
            raised.append(outcome.alert)

        duration_ms = int((time.monotonic() - wall) * 1000)
        rss = self._reclaim_memory()

        self._poll_count += 1
        return PollResult(
            started_at=started,
            finished_at=utc_now(),
            snapshot=snapshot,
            scenes=scenes,
            colonies=colonies,
            health=health,
            alerts=raised,
            degraded=tuple(dict.fromkeys(degraded)),
            ok=snapshot is not None or bool(colonies),
            duration_ms=duration_ms,
            rss_after_bytes=rss,
        )

    def _persist(
        self,
        snapshot: SpaceWeatherSnapshot | None,
        scenes: list[SarScene],
        colonies: list[Colony],
        health: list[SourceHealth],
    ) -> None:
        """Write a pass to the store, isolating each write so one failure is local.

        A failure to write health telemetry must not lose the observations, and
        vice versa, so each is committed independently.

        Parameters
        ----------
        snapshot
            Space weather snapshot, if any.
        scenes
            SAR scenes.
        colonies
            Colony records.
        health
            Health entries.
        """
        if snapshot is not None:
            try:
                self.store.record_snapshot(
                    snapshot, storm_kp=self.config.space_weather.storm_kp
                )
            except TrackerError as exc:
                _LOG.error("cannot persist space weather: %s", exc)
        if colonies:
            try:
                self.store.upsert_colonies(colonies)
            except TrackerError as exc:
                _LOG.error("cannot persist colonies: %s", exc)
        for scene in scenes:
            try:
                self.store.record_sar_scene(scene)
            except TrackerError as exc:
                _LOG.error("cannot persist scene %s: %s", scene.scene_id, exc)
        if health:
            try:
                self.store.record_health(health)
            except TrackerError as exc:
                _LOG.error("cannot persist health: %s", exc)

    def _deliver(self, rule_id: str, alert: Alert) -> None:
        """Transmit an alert and record the delivery outcome.

        Parameters
        ----------
        rule_id
            The originating rule.
        alert
            The alert to deliver.
        """
        delivered = False
        error: str | None = None
        try:
            delivered = self.dispatcher.dispatch(alert)
        except TrackerError as exc:
            error = str(exc)
            _LOG.error("alert delivery failed for %s: %s", rule_id, exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _LOG.exception("unexpected delivery error for %s", rule_id)

        try:
            self.store.record_alert(
                rule_id=rule_id,
                severity=alert.severity,
                title=alert.title,
                body=alert.body,
                context=alert.context,
                delivered=delivered,
                delivery_error=error,
            )
        except TrackerError as exc:
            _LOG.error("cannot persist alert %s: %s", rule_id, exc)

    def page_node_unhealthy(self, exc: TrackerError, failures: int) -> None:
        """Deliver the node-unhealthy alert, at most once per repeat window.

        This alert bypasses the rule engine, so it has no latch and no cooldown
        of its own, and the daemon re-raises it on every restart. A node stuck
        in a failing loop therefore re-paged on each pass: at the default
        five-minute cadence that is 288 pages a day for a single outage.

        The window is checked against the last *recorded* page, so a restart
        does not reset the count. Delivery failures are logged and swallowed --
        a node that cannot page is exactly the case where raising would only
        hide the original error behind a second one.

        Parameters
        ----------
        exc
            The failure to report.
        failures
            Consecutive failed passes.
        """
        if self.store is not None:
            try:
                last = self.store.last_alert_at("node-unhealthy")
            except StoreError:
                last = None
            if last is not None:
                elapsed = (utc_now() - last).total_seconds()
                window = self.config.daemon.unhealthy_repeat_seconds
                if elapsed < window:
                    _LOG.warning(
                        "node unhealthy after %d pass(es); already paged %.0fs ago, "
                        "withholding the repeat until the %ds window elapses",
                        failures, elapsed, window,
                    )
                    return

        alert = Alert(
            rule_id="node-unhealthy",
            severity=Severity.CRITICAL,
            title=f"Monitoring node {self.config.site.name} is failing",
            body=(
                f"{failures} consecutive polling passes have failed. The node is "
                "not monitoring anything, and any storm, lead or colony event in this "
                "window would go unreported. Check network egress, the configured "
                "upstream services, and the state directory. This repeats at most "
                f"every {self.config.daemon.unhealthy_repeat_seconds}s until recovery."
            ),
            context={"failures": failures, "error": str(exc)},
            fields=(
                ("Node", self.config.site.name),
                ("Consecutive failures", str(failures)),
                ("Last error", str(exc)[:200]),
            ),
        )
        try:
            self._deliver("node-unhealthy", alert)
        except TrackerError:
            _LOG.error("could not deliver node-unhealthy alert; the node is failing and silent")

    def _reclaim_memory(self) -> int | None:
        """Force a collection between passes and warn if over budget.

        Returns
        -------
        int | None
            RSS in bytes after the collection.

        Notes
        -----
        This runs *between* passes, never during one. Forcing a collection in
        the middle of parsing a large payload would cost more time than it saves
        in memory.
        """
        before, after = collect()
        ceiling = self.config.daemon.max_rss_bytes
        warn_at = ceiling * self.config.daemon.rss_warn_fraction
        current = after or before
        if current and current > warn_at:
            usage = current / (1024 * 1024)
            limit = ceiling / (1024 * 1024)
            if current > ceiling:
                _LOG.error(
                    "RSS %.1f MiB exceeds the configured ceiling of %.1f MiB; "
                    "check sar.grid_cells and the retention window",
                    usage, limit,
                )
            else:
                _LOG.warning(
                    "RSS %.1f MiB is above %.0f%% of the %.1f MiB budget",
                    usage, self.config.daemon.rss_warn_fraction * 100, limit,
                )
        return current

    def prune(self) -> dict[str, int]:
        """Apply the retention policy.

        Returns
        -------
        dict[str, int]
            Rows deleted per table.
        """
        deleted = self.store.prune(retention_days=self.config.paths.retention_days)
        if any(deleted.values()):
            _LOG.info("pruned: %s", {k: v for k, v in deleted.items() if v})
        return deleted


class Daemon:
    """Supervises repeated :meth:`PollEngine.run_once` calls under systemd.

    Parameters
    ----------
    engine
        The engine to drive.
    config
        The loaded configuration.
    on_result
        Optional callback invoked with each :class:`PollResult`, used by the
        CLI to render output and by tests to assert behaviour.
    sleep_fn
        Injectable sleep, so a test can drive many iterations instantly.

    Examples
    --------
    >>> from emperor_space_tracker.config import load_config
    >>> from emperor_space_tracker.store import Store
    >>> cfg = load_config(use_user_config=False)
    >>> engine = PollEngine(cfg, Store(":memory:"))
    >>> Daemon(engine, cfg, sleep_fn=lambda s: None).run(max_passes=1)  # doctest: +SKIP
    0
    """

    def __init__(
        self,
        engine: PollEngine,
        config: Config,
        *,
        on_result: Callable[[PollResult], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.engine = engine
        self.config = config
        self.on_result = on_result
        self.sleep_fn = sleep_fn
        self._stop = False
        self._signum: int | None = None
        self._failures = 0
        self._last_heartbeat = time.monotonic()

    def request_stop(self, signum: int | None = None, _frame: FrameType | None = None) -> None:
        """Ask the loop to finish the current pass and exit cleanly.

        Registered for SIGTERM and SIGINT. systemd sends SIGTERM on
        ``systemctl --user stop`` and expects a prompt exit; the handler only
        sets a flag so the in-flight poll is allowed to finish and its results
        are committed rather than discarded.

        Parameters
        ----------
        signum
            The signal number received, recorded for the exit message.
        _frame
            Unused frame from the signal handler.
        """
        if not self._stop:
            _LOG.info(
                "received %s; finishing current poll then exiting",
                signal.Signals(signum).name if signum else "stop request",
            )
        self._stop = True
        self._signum = signum

    def _stopping(self) -> bool:
        """Return whether a stop has been requested.

        The flag is flipped from a signal handler, so it changes at points the
        loop's own control flow cannot see. Reading it through a method keeps
        that asynchrony visible to a reader and stops a type checker from
        concluding, at the first ``if self._stop:`` after ``while not
        self._stop``, that the branch is permanently dead.

        Returns
        -------
        bool
            ``True`` once :meth:`request_stop` has fired.
        """
        return self._stop

    def install_signal_handlers(self) -> None:
        """Register handlers for SIGTERM, SIGINT and SIGHUP.

        SIGHUP is treated as a stop rather than a config reload. Reloading
        configuration on a remote node during polar night, where a dropped
        connection cannot be distinguished from a hung process, is a worse
        failure than a restart that systemd performs in under a second.
        """
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.request_stop)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self.request_stop)

    def run(self, *, max_passes: int | None = None) -> int:
        """Run polling passes until stopped.

        Parameters
        ----------
        max_passes
            Stop after this many passes. ``None`` runs until a signal arrives.

        Returns
        -------
        int
            Process exit code. See the module docstring for the contract.

        Raises
        ------
        TrackerError
            If the failure counter exceeds
            ``daemon.consecutive_failures_before_page``. The caller turns this
            into exit code 1; letting the exception escape keeps the decision in
            one place.
        """
        interval = self.config.daemon.poll_interval_seconds
        self._log_startup()
        passes = 0

        while not self._stopping():
            if max_passes is not None and passes >= max_passes:
                break
            passes += 1
            try:
                result = self.engine.run_once()
            except TrackerError as exc:
                self._failures += 1
                _LOG.error("poll %d failed: %s", passes, exc)
                self._maybe_page(TrackerError(str(exc)))
                if self._stop:
                    break
                self._sleep_before_retry(interval)
                continue

            if self.on_result is not None:
                self.on_result(result)

            if result.ok:
                self._failures = 0
            else:
                self._failures += 1
                self._maybe_page(
                    TrackerError("; ".join(result.degraded) or "poll produced no data")
                )

            if result.degraded:
                _LOG.warning("pass %d degraded: %s", passes, ", ".join(result.degraded))
            _LOG.info("pass %d %s", passes, result.summary())

            self._maybe_prune()
            self._maybe_heartbeat(result)

            if self._stopping() or (max_passes is not None and passes >= max_passes):
                break
            self._sleep(interval)

        if self._failures:
            _LOG.error("exiting with %d consecutive failed pass(es)", self._failures)
            return 1
        _LOG.info("daemon stopped cleanly after %d pass(es)", passes)
        return 0

    def _sleep(self, seconds: float) -> None:
        """Sleep in one-second slices so a stop request is honoured promptly.

        Sleeping the full five-minute interval in one call would make
        ``systemctl stop`` appear to hang for up to five minutes, which
        systemd's default 90 s ``TimeoutStopSec`` would escalate to SIGKILL --
        killing the process mid-poll and losing the pass.

        Parameters
        ----------
        seconds
            Total sleep requested.
        """
        deadline = time.monotonic() + seconds
        while not self._stopping():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.sleep_fn(min(1.0, remaining))

    def _sleep_before_retry(self, interval: float) -> None:
        """Back off after a failed pass.

        Parameters
        ----------
        interval
            The configured poll interval.
        """
        backoff = min(interval, 30.0 * self._failures)
        _LOG.warning("retrying in %.0fs (failure %d)", backoff, self._failures)
        self._sleep(backoff)

    def _maybe_prune(self) -> None:
        """Run retention pruning once an hour."""
        interval = max(1.0, self.config.daemon.poll_interval_seconds)
        every = max(1, int(3600 / interval))
        if self.engine.poll_count % every == 0:
            try:
                self.engine.prune()
            except TrackerError as exc:
                _LOG.warning("prune failed: %s", exc)

    def _maybe_heartbeat(self, result: PollResult) -> None:
        """Emit a periodic status line regardless of poll outcome.

        Parameters
        ----------
        result
            The result of the pass just completed.
        """
        now = time.monotonic()
        if now - self._last_heartbeat < self.config.daemon.heartbeat_seconds:
            return
        self._last_heartbeat = now
        rss = current_rss_bytes() or result.rss_after_bytes or 0
        _LOG.info(
            "heartbeat: pass %d, uptime sources ok=%d degraded=%d, rss=%.1f MiB of %.1f MiB, "
            "alerts=%d, db=%.1f MiB",
            self.engine.poll_count,
            sum(1 for h in result.health if h.ok),
            len(result.degraded),
            rss / (1024 * 1024),
            self.config.daemon.max_rss_bytes / (1024 * 1024),
            len(result.alerts),
            self.engine.store.size_bytes() / (1024 * 1024),
        )

    def _maybe_page(self, exc: TrackerError) -> None:
        """Alert when the node has been failing for too long.

        Parameters
        ----------
        exc
            The failure to report.

        Raises
        ------
        TrackerError
            If the failure count has reached the configured limit.
        """
        limit = self.config.daemon.consecutive_failures_before_page
        if self._failures < limit:
            return
        # Deliver through the engine rather than reaching into its private
        # _deliver: the repeat window and the alert construction belong with
        # the rest of the alerting, not inside the supervisor.
        self.engine.page_node_unhealthy(exc, self._failures)
        msg = f"node unhealthy after {self._failures} consecutive failures: {exc}"
        raise TrackerError(msg) from exc

    def _log_startup(self) -> None:
        """Log the effective configuration and resolved paths at startup.

        systemd journals this, so it is the first thing an operator sees when
        diagnosing why a unit is not doing what they expected.
        """
        _LOG.info(
            "emperor-space-tracker starting: node=%s, poll=%.0fs, rss_budget=%.1f MiB, "
            "space_weather=%s, sar=%s/%s, alerts=%s",
            self.config.site.name,
            self.config.daemon.poll_interval_seconds,
            self.config.daemon.max_rss_bytes / (1024 * 1024),
            "on" if self.config.space_weather.enabled else "off",
            "on" if self.config.sar.enabled else "off",
            self.config.sar.backend,
            self.dispatcher_summary(),
        )
        _LOG.debug("%s", self.config.describe())
        _LOG.info("state directory: %s", self.config.paths.resolved_state_dir())

    def dispatcher_summary(self) -> str:
        """Return a description of the configured alert channel.

        Returns
        -------
        str
        """
        return self.engine.dispatcher.describe()


def build_engine_and_daemon(
    config: Config,
    *,
    dry_run_alerts: bool = False,
    store: Store | None = None,
) -> tuple[PollEngine, Daemon]:
    """Construct the store, engine and daemon for a configuration.

    Parameters
    ----------
    config
        The loaded configuration.
    dry_run_alerts
        Force the null alert channel regardless of environment, so a threshold
        can be validated against live data without paging anyone.
    store
        An already-open store to reuse. Supplied by callers that need the
        database before building the engine -- `est status` and `est poll` both
        read around the poll, and opening a second connection to the same SQLite
        file just to throw it away leaks a handle and doubles the WAL cost.

    Returns
    -------
    tuple[PollEngine, Daemon]
        The wired engine and its supervisor. The engine does *not* own ``store``
        and has no ``close()`` of its own: closing it is the caller's
        responsibility, exactly once, via ``store.close()``.

    Raises
    ------
    TrackerError
        If the state directory or database cannot be opened.
    """
    active_store = store if store is not None else open_default_store(config)
    # One HTTP client for the whole process. Building a second one here would
    # mean a second TLS context -- measured at ~3.9 MiB resident on its own,
    # against a daemon whose entire memory budget is 48 MiB -- and a second
    # connection pool, for no benefit.
    http = HttpClient(
        timeout=config.daemon.http_timeout_seconds,
        max_retries=config.alerts.max_retries,
    )
    dispatcher = (
        NullNotifier()
        if dry_run_alerts
        else build_dispatcher(
            client=http,
            enabled=config.alerts.enabled,
            webhook_url_env=config.alerts.webhook_url_env,
            max_retries=config.alerts.max_retries,
        )
    )
    engine = PollEngine(config, active_store, http=http, dispatcher=dispatcher)
    return engine, Daemon(engine, config)


def uptime_summary(daemon_started: datetime) -> str:
    """Return a human-readable uptime string.

    Parameters
    ----------
    daemon_started
        When the process started.

    Returns
    -------
    str
    """
    delta = utc_now() - daemon_started
    seconds = int(delta.total_seconds())
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def next_poll_time(interval_seconds: float, last: datetime) -> datetime:
    """Return when the next pass is due.

    Parameters
    ----------
    interval_seconds
        Configured interval.
    last
        When the last pass started.

    Returns
    -------
    datetime
    """
    return last + timedelta(seconds=interval_seconds)
