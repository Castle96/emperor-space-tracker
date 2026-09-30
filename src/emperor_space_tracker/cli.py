"""Command line interface.

Design notes
------------
**No third-party argument parser.** ``argparse`` is in the standard library and
the daemon must not carry a dependency tree. This costs some typing boilerplate
and buys a process that installs with nothing to resolve.

**Subcommands mirror operational intent, not internals.** An operator at 3 a.m.
wants ``est doctor`` or ``est status``, not ``est sources.poll``.

**Every command that touches the network has a ``--dry-run``.** Validating a
threshold change against live data must not page anyone.

Commands
--------
``poll``       One polling pass, then exit. The cron/systemd-timer primitive.
``run``        The supervised long-running loop (the systemd service).
``status``     Last observation, health, memory, recent alerts.
``doctor``     Environment and connectivity checks, including a memory report.
``config``     Show, validate, locate or initialise configuration.
``colonies``   Show the colony census within range.
``seed``       Backfill a synthetic history so the dashboard has something to draw.
``dashboard``  Launch the Streamlit frontend.
``install``    Install and enable the systemd user units (collector + dashboard).
``uninstall``  Stop, disable and remove those units, keeping the data.
``auth``       Manage dashboard accounts and inspect the proxy secret.
``reset``      Clear stored observations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote as _url_quote

from . import __version__
from .config import Config, default_config_path, load_config, save_user_config
from .engine import PollResult, build_engine_and_daemon
from .errors import ConfigError, TrackerError
from .logging_setup import get_rss_bytes, setup_logging, shutdown_logging
from .memory import format_breakdown, measure_breakdown, peak_rss_bytes, rss_mib
from .models import Severity, utc_now
from .net import HttpClient, supports_range
from .store import open_default_store

__all__ = ["build_parser", "main"]

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
EXIT_SIGNAL: Final = 2

# ANSI colour, used only when stdout is a TTY. Redirecting to a log file or a
# journal must produce clean text.
_COLOURS: Final[dict[str, str]] = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
}


class _Style:
    """Minimal ANSI styling that degrades to plain text when piped."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *names: str) -> str:
        """Wrap ``text`` in the named colour codes if styling is enabled."""
        if not self.enabled or not names:
            return text
        prefix = "".join(_COLOURS.get(n, "") for n in names)
        return f"{prefix}{text}{_COLOURS['reset']}"


def _style() -> _Style:
    """Return a styler bound to whether stdout is a terminal."""
    return _Style(sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)


def _emit(text: str = "") -> None:
    """Print a line to stdout, flushing immediately.

    The flush matters under systemd: stdout is a journal socket, and a buffered
    process that is SIGKILLed on timeout loses the lines that explain why.
    """
    print(text, flush=True)


def _num(value: object, default: float, spec: str) -> str:
    """Format a possibly-missing number, treating only ``None`` as absent.

    Written as a helper because the obvious inline spelling is wrong in a way
    that hides itself: ``value or float("nan")`` renders a stored Kp of ``0.0``
    as ``nan``, because ``0.0`` is falsy. Kp 0.0 is the *normal* value during
    quiet conditions, so that bug printed ``Kp nan`` precisely when there was
    nothing wrong.

    Parameters
    ----------
    value
        The value to format, typically a SQLite column that may be NULL.
    default
        Substituted when ``value`` is ``None`` (not when it is zero).
    spec
        A format spec, applied to a float.

    Returns
    -------
    str
        The formatted number.
    """
    if value is None:
        return format(default, spec)
    return format(float(value), spec)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------


def _cmd_poll(args: argparse.Namespace, config: Config) -> int:
    """Run a single polling pass and render the result."""
    store = open_default_store(config)
    try:
        engine, _ = build_engine_and_daemon(
            config,
            dry_run_alerts=args.dry_run or not args.send,
            store=store,
        )
        result = engine.run_once(dry_run=args.dry_run)
    finally:
        store.close()

    if args.json:
        _emit(json.dumps(_result_to_dict(result), indent=2, default=str))
        return EXIT_OK if result.ok else EXIT_ERROR

    style = _style()
    _emit()
    _emit(style("  " + result.summary(), "bold"))
    _emit()
    _render_result(result, config, style, verbose=args.verbose)
    return EXIT_OK if result.ok else EXIT_ERROR


def _render_result(
    result: PollResult,
    config: Config,
    style: _Style,
    *,
    verbose: bool = False,
) -> None:
    """Print a human-readable rendering of a poll result.

    Parameters
    ----------
    result
        The pass outcome.
    config
        The loaded configuration, used for the memory budget display.
    style
        ANSI styler.
    verbose
        Include the full health table.
    """
    limit = 5 if verbose else 3
    _emit(f"  {'node':<18} {result.started_at.astimezone(UTC).isoformat(timespec='seconds')}")

    if result.snapshot is not None:
        _emit(f"  {'space weather':<18} {result.snapshot.headline()}")
        plasma = result.snapshot.plasma
        field = result.snapshot.magnetic_field
        kp = result.snapshot.kp
        if plasma is not None:
            _emit(
                f"  {'':<18}   plasma {_num(plasma.speed_kms, 0.0, '.0f')} km/s, "
                f"{_num(plasma.density_per_cm3, 0.0, '.1f')} p/cm3, "
                f"{_num(plasma.temperature_k, 0.0, '.0f')} K [{plasma.source}]"
            )
        if field is not None:
            _emit(
                f"  {'':<18}   field  Bt {_num(field.bt_nt, 0.0, '.1f')} nT, "
                f"Bz(GSM) {_num(field.bz_gsm_nt, 0.0, '+.1f')} nT [{field.source}]"
            )
        if kp is not None and kp.value is not None:
            kind = "nowcast" if kp.is_estimated else "3-hour"
            _emit(f"  {'':<18}   Kp     {kp.value:.1f} ({kind})")
        if result.snapshot.f107_sfu is not None:
            _emit(f"  {'':<18}   F10.7  {result.snapshot.f107_sfu:.1f} sfu")
        if result.snapshot.degraded:
            _emit(
                "  "
                + style("  DEGRADED           ", "red")
                + ", ".join(result.snapshot.degraded)
            )

    if result.colonies:
        _emit(f"  {'colonies':<18} {len(result.colonies)} within range")
        for colony in result.colonies[:limit]:
            population = f"{colony.population_estimate:,}" if colony.population_estimate else "n/a"
            ice = f"{colony.fast_ice_ratio:.0%}" if colony.fast_ice_ratio is not None else "n/a"
            _emit(
                f"  {'':<18}   {colony.name[:38]:<38} {population:>7} pairs  fast-ice {ice}"
            )

    if result.scenes:
        _emit(f"  {'SAR scenes':<18} {len(result.scenes)}")
        ranked = sorted(result.scenes, key=lambda s: s.open_water_fraction, reverse=True)
        for scene in ranked[:limit]:
            flag = style(" [SYNTHETIC]", "yellow") if scene.is_synthetic else ""
            colour = _stability_colour(scene.classify())
            _emit(
                f"  {'':<18}   {scene.colony_id[:30]:<30} "
                f"{scene.mean_db:6.2f} dB  water {scene.open_water_fraction_percent:5.1f}%  "
                f"{style(scene.classify(), colour)}{flag}"
            )

    if result.alerts:
        _emit()
        _emit(style("  alerts raised", "bold"))
        for alert in result.alerts:
            colour = _severity_colour(alert.severity)
            _emit(
                f"    {style(alert.severity.value.upper(), colour):<20} {alert.title}"
            )

    if verbose and result.health:
        _emit()
        _emit(style("  source health", "bold"))
        for entry in result.health:
            colour = {"ok": "green", "degraded": "yellow", "down": "red"}[entry.status]
            _emit(
                f"    {entry.source:<18} {style(entry.status, colour):<18} "
                f"{entry.latency_ms:>5} ms  {entry.detail}"
            )

    rss = result.rss_after_bytes or get_rss_bytes()
    if rss:
        _emit(
            f"  {'resources':<18} rss {rss_mib(rss)} MiB of "
            f"{config.effective_rss_mib():.1f} MiB budget, "
            f"pass took {result.duration_ms} ms"
        )
    _emit()


def _stability_colour(stability: str) -> str:
    """Return the display colour for a stability grade.

    Parameters
    ----------
    stability
        One of the grades from :meth:`~emperor_space_tracker.models.SarScene.classify`.

    Returns
    -------
    str
    """
    return {
        "stable": "green",
        "nominal": "cyan",
        "stressed": "yellow",
        "breached": "red",
        "dispersed": "red",
    }.get(stability, "dim")


def _severity_colour(severity: Severity) -> str:
    """Return the display colour for an alert severity.

    Parameters
    ----------
    severity
        The severity.

    Returns
    -------
    str
    """
    return {
        Severity.DEBUG: "dim",
        Severity.INFO: "cyan",
        Severity.WARNING: "yellow",
        Severity.SEVERE: "red",
        Severity.CRITICAL: "red",
    }.get(severity, "dim")


def _result_to_dict(result: PollResult) -> dict[str, Any]:
    """Render a poll result as a JSON-serialisable mapping.

    Parameters
    ----------
    result
        The result to render.

    Returns
    -------
    dict[str, Any]
    """
    return {
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "ok": result.ok,
        "degraded": list(result.degraded),
        "duration_ms": result.duration_ms,
        "rss_after_bytes": result.rss_after_bytes,
        "space_weather": {
            "speed_kms": (
                result.snapshot.plasma.speed_kms
                if result.snapshot and result.snapshot.plasma
                else None
            ),
            "bz_gsm_nt": (
                result.snapshot.magnetic_field.bz_gsm_nt
                if result.snapshot and result.snapshot.magnetic_field
                else None
            ),
            "kp": result.snapshot.kp.value if result.snapshot and result.snapshot.kp else None,
            "f107_sfu": result.snapshot.f107_sfu if result.snapshot else None,
            "degraded": list(result.snapshot.degraded) if result.snapshot else [],
        },
        "colonies": [
            {
                "colony_id": c.colony_id,
                "name": c.name,
                "latitude": c.latitude,
                "longitude": c.longitude,
                "population_estimate": c.population_estimate,
                "population_year": c.population_year,
                "fast_ice_ratio": c.fast_ice_ratio,
            }
            for c in result.colonies
        ],
        "scenes": [
            {
                "scene_id": s.scene_id,
                "colony_id": s.colony_id,
                "observed_at": s.observed_at.isoformat(),
                "mean_db": s.mean_db,
                "open_water_fraction": s.open_water_fraction,
                "stability": s.classify(),
                "synthetic": s.is_synthetic,
            }
            for s in result.scenes
        ],
        "alerts": [
            {"rule_id": a.rule_id, "severity": a.severity.value, "title": a.title}
            for a in result.alerts
        ],
        "health": [
            {
                "source": h.source,
                "ok": h.ok,
                "status": h.status,
                "latency_ms": h.latency_ms,
                "detail": h.detail,
            }
            for h in result.health
        ],
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace, config: Config) -> int:
    """Run the supervised polling loop."""
    try:
        _, daemon = build_engine_and_daemon(config, dry_run_alerts=args.dry_run)
    except TrackerError as exc:
        _emit(f"error: {exc}")
        return EXIT_ERROR

    style = _style()
    if args.max_passes:
        _emit(f"running {args.max_passes} pass(es) then exiting")
    daemon.on_result = lambda r: _emit(_format_line(r, style))
    daemon.install_signal_handlers()

    try:
        return daemon.run(max_passes=args.max_passes)
    except TrackerError as exc:
        _emit(style(f"error: {exc}", "red"))
        return EXIT_ERROR
    except KeyboardInterrupt:
        _emit("\ninterrupted")
        return EXIT_SIGNAL
    finally:
        shutdown_logging()


def _format_line(result: PollResult, style: _Style) -> str:
    """Render one poll result as a single line.

    Parameters
    ----------
    result
        The result.
    style
        ANSI styler.

    Returns
    -------
    str
    """
    return style(result.summary(), "dim")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _cmd_status(args: argparse.Namespace, config: Config) -> int:
    """Show the most recent observation, health, memory and alerts."""
    style = _style()
    try:
        store = open_default_store(config)
    except TrackerError as exc:
        _emit(style(f"cannot open store: {exc}", "red"))
        return EXIT_ERROR

    try:
        if args.json:
            payload = {
                # The same nested mapping `est config show --json` emits, not
                # describe(): that returns a formatted text block, and every
                # other field here is structured, so a string in this slot would
                # be the one field a consumer could not read programmatically.
                "config": _config_to_dict(config),
                "latest_space_weather": _jsonable(store.latest_space_weather()),
                "health": [
                    {**h, "observed_at": h["observed_at"].isoformat()}
                    for h in store.health_summary(minutes=args.hours * 60)
                ],
                "stats": {s.table: s.rows for s in store.stats()},
                "size_bytes": store.size_bytes(),
                "alerts": [a.as_dict() for a in store.recent_alerts(limit=10)],
            }
            _emit(json.dumps(payload, indent=2, default=str))
            return EXIT_OK

        _emit()
        _emit(style("  node status", "bold"))
        _emit(f"  {'config':<18} {config.source_path}")
        _emit(f"  {'database':<18} {config.paths.database_path}")
        _emit(f"  {'size':<18} {store.size_bytes() / (1024 * 1024):.2f} MiB")
        _emit(f"  {'rss now':<18} {rss_mib(get_rss_bytes())} MiB")

        latest = store.latest_space_weather()
        if latest is None:
            _emit(f"  {'observations':<18} {style('none yet - run `est poll`', 'yellow')}")
        else:
            age = (utc_now() - latest["observed_at"]).total_seconds()
            seen = latest["observed_at"].isoformat(timespec="seconds")
            _emit(f"  {'last observed':<18} {seen} ({age / 60:.0f} min ago)")
            _emit(
                f"  {'':<18}   speed {_num(latest['speed_kms'], 0.0, '.0f')} km/s, "
                f"Bz {_num(latest['bz_gsm_nt'], 0.0, '+.1f')} nT, "
                f"Kp {_num(latest['kp_value'], 0.0, '.1f')}"
            )
            if latest["degraded"]:
                _emit(
                    "  " + style(f"  {'degraded':<18} ", "yellow")
                    + ", ".join(latest["degraded"])
                )

        health = store.health_summary(minutes=args.hours * 60)
        if health:
            _emit()
            _emit(style("  source health", "bold"))
            for entry in health:
                colour = {"ok": "green", "degraded": "yellow", "down": "red"}[entry["status"]]
                _emit(
                    f"    {entry['source']:<18} {style(entry['status'], colour):<18} "
                    f"{entry['latency_ms']:>5} ms  {entry['detail']}"
                )
        else:
            _emit()
            _emit(style("  no source health recorded in the last "
                        f"{args.hours} h - has the daemon ever run?", "yellow"))

        alerts = store.recent_alerts(limit=10)
        if alerts:
            _emit()
            _emit(style("  recent alerts", "bold"))
            for alert in alerts:
                mark = style("sent", "green") if alert.delivered else style("LOGGED", "yellow")
                _emit(
                    f"    {alert.fired_at.astimezone(UTC).strftime('%Y-%m-%d %H:%M')}Z "
                    f"{style(alert.severity.value.upper(), _severity_colour(alert.severity)):<10} "
                    f"{alert.title[:62]:<62} {mark}"
                )
                if alert.delivery_error:
                    _emit(f"      {style(alert.delivery_error, 'red')}")

        if args.verbose:
            _emit()
            _emit(style("  row counts", "bold"))
            for stat in store.stats():
                if stat.rows:
                    _emit(f"    {stat!s}")
        _emit()
        return EXIT_OK
    finally:
        store.close()


def _jsonable(value: Any) -> Any:
    """Convert datetimes in a nested structure to ISO strings.

    Parameters
    ----------
    value
        Any JSON-ish value, possibly containing datetimes.

    Returns
    -------
    Any
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    """Run environment, connectivity and configuration checks."""
    style = _style()
    checks: list[tuple[str, bool, str]] = []

    _emit()
    _emit(style(f"  emperor-space-tracker {__version__} doctor", "bold"))
    _emit()

    # -- interpreter and platform ----------------------------------------
    checks.append((
        "python",
        sys.version_info >= (3, 13),
        f"{sys.version.split()[0]} on {sys.platform} (requires >= 3.13)",
    ))

    # -- config ------------------------------------------------------------
    try:
        config.validate()
        checks.append(("config", True, str(config.source_path)))
    except ConfigError as exc:
        checks.append(("config", False, str(exc).replace("\n", "; ")))

    for warning in config.warnings:
        _emit(f"  {style('note', 'yellow')} {warning}")
    _emit()

    # -- TLS trust store ---------------------------------------------------
    trust = _check_tls(config)
    checks.append(("tls trust store", *trust))

    # -- state directory ---------------------------------------------------
    try:
        state = config.paths.ensure()
        probe = state / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(("state directory", True, f"{state} is writable"))
    except (OSError, ConfigError) as exc:
        checks.append(("state directory", False, str(exc)))

    # -- database ----------------------------------------------------------
    try:
        with open_default_store(config) as store:
            stats = {s.table: s.rows for s in store.stats()}
            checks.append((
                "database",
                True,
                f"{sum(stats.values()):,} rows, {store.size_bytes() / 1024:.0f} KiB on disk",
            ))
    except TrackerError as exc:
        checks.append(("database", False, str(exc)))

    # -- alerting ----------------------------------------------------------
    checks.append(_check_alerts(config))

    # -- dashboard extras --------------------------------------------------
    checks.append(_check_optional())

    # -- SAR backend -------------------------------------------------------
    if config.sar.enabled:
        from .sources.sar import SarClient

        available, reason = SarClient(HttpClient(), config.sar).is_available()
        checks.append(("sar backend", available, f"{config.sar.backend}: {reason}"))

    # -- connectivity ------------------------------------------------------
    if not args.offline:
        client = HttpClient(timeout=config.daemon.http_timeout_seconds, max_retries=1)
        for name, url in _endpoints_to_check(config):
            ok, detail = client.ping(url, source=name)
            ranged = supports_range(url, timeout=8.0) if ok else False
            note = f"{detail}; byte-range {'supported' if ranged else 'unavailable'}"
            checks.append((f"network {name}", ok, note))

    # -- memory ------------------------------------------------------------
    rss = get_rss_bytes()
    peak = peak_rss_bytes()
    ceiling = config.effective_rss_mib()
    if rss is not None:
        pct = rss / (config.daemon.max_rss_bytes or 1) * 100
        checks.append((
            "memory now",
            rss <= config.daemon.max_rss_bytes,
            f"{rss_mib(rss)} MiB RSS ({pct:.0f}% of the {ceiling:.0f} MiB budget); "
            f"peak {rss_mib(peak)} MiB",
        ))

    if args.memory:
        _emit(style("  memory breakdown (measured in child interpreters)", "bold"))
        _emit(format_breakdown(measure_breakdown()))
        _emit()
        _emit(style("  Each row is cumulative and includes the ones above it.", "dim"))
        _emit(
            style(
                "  A hard sub-15 MiB total RSS target is not reachable in CPython: the\n"
                "  interpreter floor plus ssl/http.client/sqlite3 is ~20.5 MiB before any\n"
                "  work happens. What this project controls is the peak. A full polling\n"
                "  pass measures ~35.8 MiB against a 48 MiB ceiling, and byte-range\n"
                "  prefetching of the multi-megabyte NOAA arrays is what keeps it there\n"
                "  instead of ~90 MiB.",
                "dim",
            )
        )
        _emit()

    # -- summary -----------------------------------------------------------
    _emit(style("  checks", "bold"))
    failures = 0
    for name, ok, detail in checks:
        mark = style("PASS", "green") if ok else style("FAIL", "red")
        _emit(f"    {mark}  {name:<22} {style(detail, 'dim')}")
        if not ok:
            failures += 1
    _emit()
    if failures:
        _emit(style(f"  {failures} check(s) failed", "red"))
    else:
        _emit(style("  all checks passed", "green"))
    _emit()
    return EXIT_OK if failures == 0 else EXIT_ERROR


def _check_tls(config: Config) -> tuple[bool, str]:
    """Verify that a usable CA trust store is present.

    Parameters
    ----------
    config
        The loaded configuration.

    Returns
    -------
    tuple[bool, str]
        ``(ok, detail)``.
    """
    import ssl

    try:
        paths = ssl.get_default_verify_paths()
    except Exception as exc:
        return False, f"cannot query the default trust store: {exc}"
    if paths.cafile and Path(paths.cafile).is_file():
        return True, f"CA bundle present at {paths.cafile}"
    if paths.capath and Path(paths.capath).is_dir():
        return True, f"CA directory present at {paths.capath}"
    return False, (
        "no system CA bundle found. Install your distribution's ca-certificates "
        "package (apt install ca-certificates / dnf install ca-certificates / "
        "apk add ca-certificates) or TLS verification will fail."
    )


def _check_alerts(config: Config) -> tuple[str, bool, str]:
    """Report the configured alert channel.

    Parameters
    ----------
    config
        The loaded configuration.

    Returns
    -------
    tuple[str, bool, str]
        ``(name, ok, detail)``. A missing webhook is reported as a pass with a
        note rather than a failure: a node with local-only alerting is a valid,
        supported configuration.
    """
    from .alerts.notifier import resolve_webhook

    if not config.alerts.enabled:
        return ("alerts", True, "disabled in configuration; alerts are logged only")
    url = resolve_webhook(config.alerts.webhook_url_env)
    if url is None:
        return (
            "alerts",
            True,
            f"${config.alerts.webhook_url_env} is not set; alerts are logged to "
            f"{config.paths.log_path} and shown in the dashboard",
        )
    from .alerts.notifier import _redact_url

    return ("alerts", True, f"discord webhook configured at {_redact_url(url)}")


def _check_optional() -> tuple[str, bool, str]:
    """Report availability of the optional frontend stack.

    Returns
    -------
    tuple[str, bool, str]
    """
    from . import DASHBOARD_AVAILABLE

    if DASHBOARD_AVAILABLE:
        return ("dashboard", True, "streamlit and plotly are installed")
    return (
        "dashboard",
        True,
        "not installed; the daemon does not need them. "
        "Run `uv sync --extra dashboard` to enable `est dashboard`",
    )


def _endpoints_to_check(config: Config) -> list[tuple[str, str]]:
    """Return the endpoints ``doctor`` probes.

    Parameters
    ----------
    config
        The loaded configuration, used to probe the taxon the node actually
        tracks rather than a hardcoded one.

    Returns
    -------
    list[tuple[str, str]]
        ``(label, url)`` pairs.
    """
    from .sources.biological import GBIF_ENDPOINTS
    from .sources.space_weather import SWPC_ENDPOINTS

    return [
        ("swpc kp", SWPC_ENDPOINTS["planetary_k_index"]),
        ("swpc plasma", SWPC_ENDPOINTS["plasma"]),
        # The species actually configured, not a literal. Probing a hardcoded
        # taxon reported a healthy node for a species this node does not track,
        # and reported nothing at all when the configured name failed to resolve
        # -- which is the failure an operator most needs to see.
        (
            f"gbif ({config.colonies.species})",
            f"{GBIF_ENDPOINTS['species_match']}?name={_url_quote(config.colonies.species)}",
        ),
    ]


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def _cmd_config(args: argparse.Namespace, config: Config) -> int:
    """Show, locate or initialise the configuration."""
    style = _style()
    if args.action == "path":
        _emit(str(config.source_path or default_config_path()))
        return EXIT_OK
    if args.action == "init":
        try:
            path = save_user_config(Path(args.out) if args.out else None, overwrite=args.force)
        except ConfigError as exc:
            _emit(style(f"error: {exc}", "red"))
            return EXIT_ERROR
        _emit(f"wrote {path}")
        _emit(f"edit it, then run: {sys.argv[0]} doctor")
        return EXIT_OK
    if args.action == "show":
        if args.json:
            _emit(json.dumps(_config_to_dict(config), indent=2, default=str))
        else:
            _emit(config.describe())
        return EXIT_OK
    if args.action == "validate":
        try:
            config.validate()
        except ConfigError as exc:
            _emit(style(str(exc), "red"))
            return EXIT_ERROR
        _emit(style("configuration is valid", "green"))
        return EXIT_OK
    return EXIT_ERROR


def _config_to_dict(config: Config) -> dict[str, Any]:
    """Render a configuration as a nested mapping.

    Parameters
    ----------
    config
        The configuration.

    Returns
    -------
    dict[str, Any]
    """
    from dataclasses import asdict

    return {
        "source_path": str(config.source_path) if config.source_path else None,
        "site": asdict(config.site),
        "paths": asdict(config.paths),
        "daemon": asdict(config.daemon),
        "space_weather": asdict(config.space_weather),
        "sar": asdict(config.sar),
        "colonies": asdict(config.colonies),
        "alerts": asdict(config.alerts),
        "dashboard": asdict(config.dashboard),
        "logging": asdict(config.logging),
    }


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------


def _auth_store_path(config: Config) -> Path:
    """Return the auth database path.

    Deliberately beside the monitoring database rather than inside it, so a
    backup of observations contains no password hashes and the dashboard never
    needs write access to the store it only reads.

    Parameters
    ----------
    config
        The loaded configuration.

    Returns
    -------
    Path
    """
    return config.paths.resolved_state_dir() / "auth.sqlite3"


def _read_password(prompt: str) -> str:
    """Prompt for a password twice, without echoing.

    Read from the terminal rather than ``argv`` so it never appears in ``ps``,
    in a shell history, or in a systemd journal.

    Parameters
    ----------
    prompt
        Label shown to the operator.

    Returns
    -------
    str
        The confirmed password.

    Raises
    ------
    TrackerError
        If the two entries differ, or neither was entered.
    """
    import getpass

    while True:
        first = getpass.getpass(f"{prompt}: ")
        if not first:
            msg = "password must not be empty"
            raise TrackerError(msg)
        second = getpass.getpass(f"{prompt} (confirm): ")
        if first == second:
            return first
        _emit("  passwords did not match; try again")


def _cmd_auth(args: argparse.Namespace, config: Config) -> int:
    """Manage dashboard accounts.

    Passwords are prompted for interactively and never accepted as an
    argument, because an argument is visible in ``ps`` output to every user on
    the machine and lands in shell history.
    """
    from .auth import AuthStore, Role
    from .errors import StoreError as _StoreError

    style = _style()
    path = _auth_store_path(config)

    if args.auth_action == "add-user":
        if not args.password:
            _emit(
                style(
                    "  refusing to set an empty password; pass a prompt, not --password",
                    "red",
                )
            )
            return EXIT_ERROR
        try:
            with AuthStore(path) as store:
                if store.count_active() == 0 and args.confirm is False:
                    _emit(
                        style(
                            "  refusing: --confirm is required for the first account",
                            "red",
                        )
                    )
                    return EXIT_ERROR
                user = store.add_user(
                    args.username, _read_password("password"), role=Role(args.role)
                )
        except TrackerError as exc:
            _emit(style(f"error: {exc}", "red"))
            return EXIT_ERROR
        except _StoreError as exc:
            _emit(style(f"error: {exc}", "red"))
            return EXIT_ERROR
        _emit(style(f"created user {user.username} with role {user.role}", "green"))
        _emit(f"  {path}")
        return EXIT_OK

    if args.auth_action == "list":
        try:
            with AuthStore(path) as store:
                users = store.users()
        except _StoreError as exc:
            _emit(style(f"error: {exc}", "red"))
            return EXIT_ERROR
        if not users:
            _emit(style("  no dashboard accounts configured", "yellow"))
            _emit(f"  create one with: {sys.argv[0]} auth add-user <name> --confirm")
            return EXIT_OK
        _emit(f"  {'user':<24} {'role':<7} {'state':<9} created")
        _emit(f"  {'-' * 24} {'-' * 7} {'-' * 9} {'-' * 10}")
        for user in users:
            state = "disabled" if user.disabled_at else "active"
            _emit(f"  {user.username:<24} {user.role:<7} {state:<9} {user.created_at:%Y-%m-%d}")
        return EXIT_OK

    if args.auth_action in {"set-role", "disable", "remove"}:
        try:
            with AuthStore(path) as store:
                if args.auth_action == "set-role":
                    store.set_role(args.username, Role(args.role))
                    _emit(style(f"  {args.username} is now {args.role}", "green"))
                elif args.auth_action == "disable":
                    store.disable(args.username)
                    _emit(style(f"  {args.username} disabled", "green"))
                else:
                    store.remove(args.username)
                    _emit(style(f"  {args.username} removed", "green"))
        except (TrackerError, _StoreError) as exc:
            _emit(style(f"error: {exc}", "red"))
            return EXIT_ERROR
        return EXIT_OK

    if args.auth_action == "passwd":
        try:
            with AuthStore(path) as store:
                if not args.password:
                    _emit(
                        style(
                            "  refusing to set an empty password; pass a prompt, not --password",
                            "red",
                        )
                    )
                    return EXIT_ERROR
                store.set_password(args.username, _read_password("new password"))
        except (TrackerError, _StoreError) as exc:
            _emit(style(f"error: {exc}", "red"))
            return EXIT_ERROR
        _emit(style(f"  password changed for {args.username}", "green"))
        return EXIT_OK

    if args.auth_action == "show-secret":
        # Reports whether the proxy secret is *reachable*, never its value.
        from .deploy import dashboard_env_file

        secret = os.environ.get(config.dashboard.auth.proxy_secret_env)
        env_name = config.dashboard.auth.proxy_secret_env
        env_file = dashboard_env_file()
        file_secret = None
        if env_file.is_file():
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line.startswith(f"{env_name}="):
                    file_secret = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if secret:
            _emit(style(f"  ${env_name} is set in this process's environment", "green"))
        elif file_secret:
            _emit(style(f"  ${env_name} is set in {env_file}", "green"))
            _emit(
                style(
                    "        the dashboard service reads it from there; this shell does not",
                    "dim",
                )
            )
        else:
            _emit(style(f"  ${env_name} is not set anywhere the dashboard can see", "red"))
            _emit("  the dashboard will refuse every proxied identity while it is unset")
            _emit(f"  create {env_file} with mode 0600, containing:")
            _emit(f"    {env_name}=$(openssl rand -base64 32)")
            return EXIT_ERROR
        _emit(style("  its value is deliberately not printed", "dim"))
        return EXIT_OK

    return EXIT_ERROR


# ---------------------------------------------------------------------------
# colonies
# ---------------------------------------------------------------------------


def _cmd_colonies(args: argparse.Namespace, config: Config) -> int:
    """List the colony census within range of the configured site."""
    from .sources.biological import ColonyCensusClient

    style = _style()
    client = HttpClient(timeout=config.daemon.http_timeout_seconds, max_retries=1)
    census = ColonyCensusClient(
        client,
        species=config.colonies.species,
        max_distance_km=config.colonies.max_distance_km,
        max_colonies=config.colonies.max_colonies,
    )
    try:
        colonies, health = census.poll(
            latitude=config.site.latitude,
            longitude=config.site.longitude,
            include_gbif=not args.catalogue_only,
        )
    except TrackerError as exc:
        _emit(style(f"error: {exc}", "red"))
        return EXIT_ERROR

    if args.json:
        _emit(json.dumps([c.as_geojson() for c in colonies], indent=2, default=str))
        return EXIT_OK

    _emit()
    _emit(style(f"  {config.colonies.species} within {config.colonies.max_distance_km:g} km of "
                f"{config.site.name}", "bold"))
    _emit(style("  census source: SCAR CEMP / CCAMLR ecosystem monitoring", "dim"))
    _emit()
    species_seen: dict[str, str] = {}
    for colony in colonies:
        species_seen.setdefault(colony.species, colony.common_name)
    for sci, common in species_seen.items():
        label = f"{common} ({sci})" if common else sci
        kinds = sorted({c.breeding_habitat for c in colonies if c.species == sci})
        _emit(style(f"  {label}", "cyan"))
        for kind in kinds:
            suffix = (
                "fast-ice SAR applies" if kind == "fast_ice" else "fast-ice SAR not applicable"
            )
            _emit(style(f"    {kind} - {suffix}", "dim"))
    _emit()
    _emit(
        f"  {'colony':<34} {'pairs':>7} {'year':>5} {'fast-ice':>9}  {'habitat':<9} species"
    )
    _emit(f"  {'-' * 34} {'-' * 7} {'-' * 5} {'-' * 9}  {'-' * 9} {'-' * 22}")
    for colony in colonies:
        population = f"{colony.population_estimate:,}" if colony.population_estimate else "n/a"
        year = str(colony.population_year) if colony.population_year else "-"
        # Only a fast-ice breeder has an ice figure; printing n/a for the other
        # kind is honest, printing 0% would not be.
        ice = (
            f"{colony.fast_ice_ratio:.0%}"
            if colony.monitors_fast_ice and colony.fast_ice_ratio is not None
            else "n/a"
        )
        _emit(
            f"  {colony.name[:34]:<34} {population:>7} {year:>5} {ice:>9}  "
            f"{colony.breeding_habitat!s:<9} {colony.species[:22]}"
        )
    _emit()
    for entry in health:
        colour = {"ok": "green", "degraded": "yellow", "down": "red"}[entry.status]
        _emit(f"  {style(entry.source, 'dim')} {style(entry.status, colour)}  {entry.detail}")
    _emit()
    return EXIT_OK


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def _cmd_seed(args: argparse.Namespace, config: Config) -> int:
    """Backfill synthetic history so the dashboard has data to render."""
    from .sources.biological import ColonyCensusClient
    from .sources.sar import SarClient

    style = _style()
    store = open_default_store(config)
    try:
        client = HttpClient(timeout=config.daemon.http_timeout_seconds, max_retries=1)
        census = ColonyCensusClient(
            client,
            species=config.colonies.species,
            max_distance_km=config.colonies.max_distance_km,
            max_colonies=config.colonies.max_colonies,
        )
        try:
            colonies, _ = census.poll(
                latitude=config.site.latitude,
                longitude=config.site.longitude,
                include_gbif=False,
            )
        except TrackerError as exc:
            _emit(style(f"error resolving colonies: {exc}", "red"))
            return EXIT_ERROR

        if not colonies:
            _emit(style("no colonies in range; nothing to seed", "yellow"))
            return EXIT_ERROR

        store.upsert_colonies(colonies)
        _emit(f"seeded {len(colonies)} colonies")

        sar_client = SarClient(client, config.sar)
        # Widen the backfill window so a season's arc is visible in the charts.
        original = config.sar.lookback_days
        config.sar.lookback_days = max(original, args.days)
        try:
            scenes, _ = sar_client.poll(
                colonies=colonies[: args.colonies], scenes_per_colony=args.days
            )
        finally:
            config.sar.lookback_days = original

        cells = 0
        for scene in scenes:
            cells += store.record_sar_scene(scene)
        _emit(f"seeded {len(scenes)} SAR scenes ({cells:,} backscatter cells)")

        now = utc_now()
        for index in range(args.passes):
            snapshot = _synthetic_snapshot(now - timedelta(minutes=5 * index), index, config)
            if snapshot is not None:
                store.record_snapshot(
                    snapshot, storm_kp=config.space_weather.storm_kp
                )
        _emit(f"seeded {args.passes} space weather snapshot(s)")

        _emit()
        _emit(style("  all seeded data is marked synthetic in provenance", "yellow"))
        _emit(f"  launch the dashboard with: {sys.argv[0]} dashboard")
        _emit()
        return EXIT_OK
    finally:
        store.close()


def _synthetic_snapshot(
    observed_at: datetime,
    index: int,
    config: Config,
) -> Any:
    """Build a plausible space weather snapshot for dashboard seeding.

    Parameters
    ----------
    observed_at
        Timestamp to stamp on the snapshot.
    index
        Pass ordinal, used to vary the values.
    config
        The loaded configuration.

    Returns
    -------
    SpaceWeatherSnapshot | None
    """
    import math
    import random

    from .models import InterplanetaryMagneticField, KpIndex, SpaceWeatherSnapshot

    # demo seed data, not cryptographic
    rng = random.Random(1000 + index)
    wave = math.sin(index / 9.0)
    speed = 380.0 + 120.0 * wave + rng.uniform(-20, 20)
    bz = -3.0 + 5.0 * wave + rng.uniform(-1, 1)
    kp = max(0.0, 2.0 + 2.5 * wave + rng.uniform(-0.4, 0.4))

    return SpaceWeatherSnapshot(
        observed_at=observed_at,
        plasma=_synthetic_plasma(observed_at, speed, rng),
        magnetic_field=InterplanetaryMagneticField(
            observed_at=observed_at,
            source="SEED",
            bt_nt=abs(bz) + rng.uniform(3, 6),
            bz_gsm_nt=bz,
            by_gsm_nt=rng.uniform(-2, 2),
            density_pct=None,
            active=True,
            provenance="synthetic.seed",
        ),
        kp=KpIndex(observed_at=observed_at, kp_index=int(kp), estimated_kp=kp,
                    provenance="synthetic.seed"),
        f107_sfu=rng.uniform(90, 160),
    )


def _synthetic_plasma(observed_at: datetime, speed: float, rng: Any) -> Any:
    """Build a synthetic plasma sample.

    Parameters
    ----------
    observed_at
        Timestamp to stamp on the sample.
    speed
        Bulk speed in km/s.
    rng
        A seeded random generator.

    Returns
    -------
    SolarWindPlasma
    """
    from .models import SolarWindPlasma

    return SolarWindPlasma(
        observed_at=observed_at,
        source="SEED",
        speed_kms=speed,
        density_per_cm3=rng.uniform(2, 12),
        temperature_k=rng.uniform(40_000, 220_000),
        active=True,
        quality=0,
        provenance="synthetic.seed",
    )


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------


def _cmd_dashboard(args: argparse.Namespace, config: Config) -> int:
    """Launch the Streamlit frontend.

    The dashboard script only produces widgets when it is executed *by* the
    Streamlit server. Importing and calling :func:`dashboard.app.main` directly
    -- which is what this used to do -- runs it in Streamlit's bare mode, where
    every call is a no-op that writes a warning to stderr, so the process printed
    a deprecation banner and exited without ever binding a port. The command has
    to start a real server instead, which is also what makes ``--port``,
    ``--host``, ``--open`` and ``--headless`` mean anything.

    The child inherits this process's environment, so ``EST_CONFIG`` and the
    ``EST_*`` overrides reach the dashboard script the same way they reach the
    daemon. The config path is passed explicitly as well so that an explicit
    ``--config`` survives the exec, rather than being silently dropped.
    """
    from . import DASHBOARD_AVAILABLE

    style = _style()
    if not DASHBOARD_AVAILABLE:
        _emit(style("the dashboard extra is not installed", "red"))
        _emit("")
        _emit("  install it with:  uv sync --extra dashboard")
        _emit("  or with pip:     pip install 'emperor-space-tracker[dashboard]'")
        _emit("")
        _emit("  the daemon does not need this; only `est dashboard` does.")
        _emit("")
        return EXIT_ERROR

    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    if not app_path.is_file():
        _emit(style(f"dashboard script missing at {app_path}", "red"))
        return EXIT_ERROR

    # Resolve flag-over-config-over-default. The flags default to None precisely
    # so that omitting one cannot quietly replace a configured bind address with
    # a built-in default.
    host = args.host if args.host is not None else config.dashboard.host
    port = args.port if args.port is not None else config.dashboard.port
    headless = config.dashboard.headless if args.headless is None else args.headless

    server = [
        f"--server.port={port}",
        f"--server.address={host}",
        # `headless` and "open a browser" are one knob: Streamlit opens a
        # browser exactly when it is not headless.
        f"--server.headless={'true' if headless else 'false'}",
        "--browser.gatherUsageStats=false",
    ]

    if headless:
        _emit(style(f"  dashboard on http://{host}:{port} (ctrl-c to stop)", "cyan"))
    else:
        _emit(style(f"  dashboard on http://{host}:{port} (opening a browser)", "cyan"))
    if host not in {"127.0.0.1", "::1", "localhost"} and not config.dashboard.auth.enabled:
        _emit(style("  note: the dashboard has no authentication", "yellow"))
        _emit(
            style(
                "        anything that can reach this port can read the store. Bind a "
                "specific",
                "dim",
            )
        )
        _emit(
            style(
                "        private address, or use an SSH tunnel, if the network is shared.",
                "dim",
            )
        )
    _emit("")

    argv = [sys.executable, "-m", "streamlit", "run", str(app_path), *server]
    child_env = dict(os.environ)
    if args.config:
        child_env["EST_CONFIG"] = str(args.config)
    if args.no_user_config:
        child_env["EST_NO_USER_CONFIG"] = "1"

    if os.name == "posix":
        # Replace this process with Streamlit rather than forking it.
        #
        # Under a systemd unit a wrapper that waits on a child is a signal
        # hazard: SIGTERM goes to the parent, the parent dies, and the child can
        # survive holding the port, so the next start fails with EADDRINUSE and
        # the unit restart-loops. `execve` makes the process *be* Streamlit, so
        # the signal lands where it matters and there is no wrapper at all.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.execve(sys.executable, argv, child_env)  # noqa: S606
        except OSError as exc:
            _emit(style(f"could not start streamlit: {exc}", "red"))
            return EXIT_ERROR
        # Unreachable unless execve somehow returns; fall through to the
        # subprocess path so behaviour is still correct on an exotic platform.

    try:
        return subprocess.call(argv, env=child_env)  # noqa: S603
    except KeyboardInterrupt:
        _emit("")
        return EXIT_OK
    except OSError as exc:
        _emit(style(f"could not start streamlit: {exc}", "red"))
        return EXIT_ERROR


# ---------------------------------------------------------------------------
# install / reset
# ---------------------------------------------------------------------------


def _cmd_install(args: argparse.Namespace, config: Config) -> int:
    """Install and enable the systemd user units.

    Writes the collector unit and, unless `--no-dashboard`, the dashboard unit
    as well. They are separate units on purpose: the collector is what must be
    up, the dashboard is a view onto it, and either can be stopped without
    affecting the other.
    """
    style = _style()
    try:
        from .deploy import (
            DASHBOARD_UNIT_NAME,
            UNIT_NAME,
            install_dashboard_unit,
            install_user_unit,
            systemctl,
            systemd_available,
        )
    except ImportError as exc:
        _emit(style(f"deploy helpers are not packaged: {exc}", "red"))
        return EXIT_ERROR

    # Explain an unusable session before writing files, so the failure is
    # "systemd is not available here" rather than a unit that silently never
    # starts.
    available, why = systemd_available()
    if not available:
        _emit(style(f"systemd user session unavailable: {why}", "yellow"))
        _emit(style("  the unit can still be written, but not enabled or started", "dim"))
        _emit(style("  containers and WSL need: loginctl enable-linger $USER", "dim"))
        _emit("")

    try:
        unit_file = install_user_unit(
            venv_python=Path(sys.executable), config_path=config.source_path
        )
        _emit(f"wrote {unit_file}")

        dash_file = None
        if config.dashboard.enabled and not args.no_dashboard:
            dash_file = install_dashboard_unit(
                venv_python=Path(sys.executable), config_path=config.source_path
            )
            _emit(f"wrote {dash_file}")
        elif args.no_dashboard:
            _emit(style("  dashboard unit skipped (--no-dashboard)", "dim"))
        else:
            _emit(style("  dashboard unit skipped ([dashboard] enabled = false)", "dim"))
    except TrackerError as exc:
        _emit(style(f"error: {exc}", "red"))
        return EXIT_ERROR

    if not available:
        _emit("")
        _emit(style("units written; enable them manually:", "yellow"))
        _emit(f"  systemctl --user enable --now {UNIT_NAME}")
        if dash_file is not None:
            _emit(f"  systemctl --user enable --now {DASHBOARD_UNIT_NAME}")
        return EXIT_OK

    if args.no_enable:
        _emit(f"skipping enable; run: systemctl --user enable --now {UNIT_NAME}")
        return EXIT_OK

    ok, output = systemctl("enable", "--now", UNIT_NAME)
    if not ok:
        _emit(style(f"could not enable {UNIT_NAME}: {output}", "red"))
        _emit(f"the unit is installed; start it with: systemctl --user start {UNIT_NAME}")
        return EXIT_ERROR

    if dash_file is not None:
        ok, output = systemctl("enable", "--now", DASHBOARD_UNIT_NAME)
        if not ok:
            # Not fatal: the collector is the thing that has to be running, and
            # a dashboard that will not start should not fail the install.
            _emit(style(f"could not enable {DASHBOARD_UNIT_NAME}: {output}", "yellow"))
            _emit(style("  the collector is running; the dashboard can be started with:", "dim"))
            _emit(style(f"    systemctl --user start {DASHBOARD_UNIT_NAME}", "dim"))
        else:
            _emit(style(f"{DASHBOARD_UNIT_NAME} installed and started", "green"))

    _emit(style(f"{UNIT_NAME} installed and started", "green"))
    _emit("")
    _emit(f"  follow the log:   journalctl --user -u {UNIT_NAME} -f")
    _emit(f"  check status:     systemctl --user status {UNIT_NAME}")
    _emit("  one-shot poll:    est poll")
    if dash_file is not None:
        host = config.dashboard.host
        # A wildcard bind has no single address to hand the operator, so show
        # loopback and let the real address come from the interface list.
        shown = "127.0.0.1" if host in {"0.0.0.0", "::"} else host  # noqa: S104
        _emit("")
        _emit(f"  dashboard:        http://{shown}:{config.dashboard.port}")
        if config.dashboard.bind_is_not_loopback and not config.dashboard.auth.enabled:
            _emit(
                style(
                    "  note: the dashboard has no authentication; any host that can "
                    "reach it",
                    "yellow",
                )
            )
            _emit(
                style(
                    "        can read the store. See the unit's header comment.",
                    "yellow",
                )
            )
        elif config.dashboard.bind_is_not_loopback and config.dashboard.auth.enabled:
            # The interesting case, and easy to leave silent by accident: the
            # port is still reachable directly, it just will not serve anyone who
            # did not come through the proxy.
            _emit(
                style(
                    "  reachable directly on the tailnet, but a request without the "
                    "proxy's signed",
                    "dim",
                )
            )
            _emit(
                style(
                    "  identity is refused. Only the proxy's authenticated users get "
                    "through.",
                    "dim",
                )
            )
    _emit("")
    return EXIT_OK


def _cmd_uninstall(args: argparse.Namespace, config: Config) -> int:
    """Stop, disable and remove the systemd user units.

    Leaves the state directory and the database alone: this removes the
    *service*, not the observations. Use ``est reset --data`` for those.
    """
    style = _style()
    try:
        from .deploy import (
            DASHBOARD_UNIT_NAME,
            UNIT_NAME,
            dashboard_unit_path,
            systemctl,
            unit_path,
        )
    except ImportError as exc:
        _emit(style(f"deploy helpers are not packaged: {exc}", "red"))
        return EXIT_ERROR

    for name, path in (
        (DASHBOARD_UNIT_NAME, dashboard_unit_path()),
        (UNIT_NAME, unit_path()),
    ):
        ok, output = systemctl("disable", "--now", name)
        if not ok:
            # An already-absent unit is a success for teardown purposes.
            _emit(style(f"  {name}: not enabled ({output.strip() or 'no unit'})", "dim"))
        else:
            _emit(style(f"  stopped and disabled {name}", "green"))
        try:
            path.unlink()
            _emit(f"  removed {path}")
        except FileNotFoundError:
            pass
        except OSError as exc:
            _emit(style(f"  could not remove {path}: {exc}", "yellow"))
            return EXIT_ERROR

    ok, output = systemctl("daemon-reload")
    if not ok:
        _emit(style(f"  daemon-reload failed: {output}", "yellow"))

    _emit("")
    _emit(style("units removed", "green"))
    _emit(f"  data left in place: {config.paths.database_path}")
    _emit("  to remove that too:  est reset --data")
    _emit("")
    return EXIT_OK


def _cmd_reset(args: argparse.Namespace, config: Config) -> int:
    """Delete stored observations."""
    style = _style()
    store = open_default_store(config)
    try:
        before = {s.table: s.rows for s in store.stats()}
        if args.data:
            store.purge()
            _emit("deleted all stored observations")
        if args.logs:
            log_path = config.paths.log_path
            for candidate in sorted(config.paths.resolved_state_dir().glob("tracker.log*")):
                try:
                    candidate.unlink()
                    _emit(f"removed {candidate}")
                except OSError as exc:
                    _emit(style(f"could not remove {candidate}: {exc}", "red"))
            if not any(config.paths.resolved_state_dir().glob("tracker.log*")):
                _emit(f"no log files present under {log_path.parent}")
        if not args.data and not args.logs:
            _emit("nothing to do; pass --data and/or --logs")
            return EXIT_ERROR
        if args.data:
            _emit("row counts before: " + json.dumps({k: v for k, v in before.items() if v}))
        return EXIT_OK
    finally:
        store.close()


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _global_flags() -> argparse.ArgumentParser:
    """Build the parser holding flags accepted both before and after the command.

    ``est poll -v`` is what everyone types; ``est -v poll`` is what people try
    when they assume flags are global. Rather than document the surprise, the
    same flag set is attached to every subcommand via ``parents=[...]``.

    Those flags default to :data:`argparse.SUPPRESS` so that an omitted flag
    leaves the attribute unset instead of writing a default. That is the only
    way both orders survive: with ordinary defaults, ``est -v poll`` parses
    ``-v`` at the top level, then the subparser's ``verbose=False`` default
    overwrites it on the way through, and the flag silently does nothing.
    :func:`_apply_global_defaults` restores the real values afterwards.

    Returns
    -------
    argparse.ArgumentParser
        An empty parser carrying only the shared flags.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "-c", "--config", metavar="PATH",
        default=argparse.SUPPRESS,
        help="configuration file to layer on top of the defaults",
    )
    shared.add_argument(
        "-v", "--verbose", action="store_true",
        default=argparse.SUPPRESS,
        help="log at DEBUG level",
    )
    shared.add_argument(
        "-q", "--quiet", action="store_true",
        default=argparse.SUPPRESS,
        help="log to file only; suppress the stderr stream",
    )
    shared.add_argument(
        "--no-user-config", action="store_true",
        default=argparse.SUPPRESS,
        help="ignore ~/.config entirely and use the packaged defaults",
    )
    return shared

#: Values for shared flags that were not given on either side of the command.
_GLOBAL_DEFAULTS: Final[dict[str, Any]] = {
    "config": None,
    "verbose": False,
    "quiet": False,
    "no_user_config": False,
}


def _apply_global_defaults(args: argparse.Namespace) -> None:
    """Fill in shared flags the parser left unset, in place.

    Parameters
    ----------
    args
        The namespace returned by :meth:`argparse.ArgumentParser.parse_args`.
    """
    for name, default in _GLOBAL_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The fully configured parser.
    """
    shared = _global_flags()
    parser = argparse.ArgumentParser(
        prog="est",
        description=(
            "Edge GIS monitoring node for Emperor penguin colonies and Antarctic "
            "fast-ice stability during polar night."
        ),
        epilog=(
            "Configuration is layered: packaged defaults, then "
            "~/.config/emperor-space-tracker/config.toml, then EST_* environment "
            "overrides. See `est config init`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"emperor-space-tracker {__version__}"
    )
    parser.add_argument(
        "-c", "--config", metavar="PATH",
        help="configuration file to layer on top of the defaults",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="log at DEBUG level",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="log to file only; suppress the stderr stream",
    )
    parser.add_argument(
        "--no-user-config", action="store_true",
        help="ignore ~/.config entirely and use the packaged defaults",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Add a subcommand that accepts the shared global flags."""
        return sub.add_parser(name, parents=[shared], **kwargs)

    # poll
    poll = add("poll", help="run one polling pass and exit")
    poll.add_argument("--json", action="store_true", help="emit the result as JSON")
    poll.add_argument("--dry-run", action="store_true",
                      help="evaluate and log alerts but do not transmit or persist")
    poll.add_argument("--send", action="store_true",
                      help="actually transmit alerts (default is to log only)")
    poll.set_defaults(func=_cmd_poll)

    # run
    run = add("run", help="run the supervised polling loop (the service)")
    run.add_argument("--max-passes", type=int, default=None, metavar="N",
                     help="stop after N passes instead of running until signalled")
    run.add_argument("--dry-run", action="store_true", help="never transmit alerts")
    run.set_defaults(func=_cmd_run)

    # status
    status = add("status", help="show the latest observation and node health")
    status.add_argument("--json", action="store_true", help="emit as JSON")
    status.add_argument("--hours", type=int, default=24, metavar="H",
                        help="health window in hours (default 24)")
    status.set_defaults(func=_cmd_status)

    # doctor
    doctor = add("doctor", help="check the environment, TLS and connectivity")
    doctor.add_argument("--offline", action="store_true", help="skip all network checks")
    doctor.add_argument("--memory", action="store_true",
                        help="include the measured import-cost breakdown")
    doctor.set_defaults(func=_cmd_doctor)

    # config
    cfg = add("config", help="inspect or initialise the configuration")
    cfg.add_argument(
        "action", choices=("show", "path", "init", "validate"),
        help="show the effective config, print its path, write a starter file, or validate",
    )
    cfg.add_argument("--json", action="store_true", help="with 'show', emit JSON")
    cfg.add_argument("--out", metavar="PATH", help="with 'init', write to this path")
    cfg.add_argument("--force", action="store_true", help="with 'init', overwrite an existing file")
    cfg.set_defaults(func=_cmd_config)

    # colonies
    colonies = add("colonies", help="list the colony census within range")
    colonies.add_argument("--json", action="store_true", help="emit GeoJSON")
    colonies.add_argument("--catalogue-only", action="store_true",
                          help="skip the GBIF scan and use the packaged census only")
    colonies.set_defaults(func=_cmd_colonies)

    # seed
    seed = add("seed", help="backfill synthetic history for the dashboard")
    seed.add_argument("--days", type=int, default=14, metavar="N",
                      help="how many acquisitions per colony to backfill")
    seed.add_argument("--colonies", type=int, default=4, metavar="N",
                      help="how many colonies to image")
    seed.add_argument("--passes", type=int, default=72, metavar="N",
                      help="how many space weather snapshots to backfill")
    seed.set_defaults(func=_cmd_seed)

    # dashboard
    dash = add("dashboard", help="launch the Streamlit frontend")
    # Every flag defaults to None so the effective value can come from
    # `[dashboard]` in the config. A flag left off the command line must not
    # silently override an operator's configured bind address with a built-in
    # default, which is how a node set up for tailnet access ends up
    # loopback-only again after a routine `--host` omission.
    dash.add_argument("--port", type=int, default=None, metavar="PORT",
                      help="listen port (default: [dashboard] port, else 8501)")
    dash.add_argument("--host", default=None, metavar="ADDR",
                      help="bind address (default: [dashboard] host, else 127.0.0.1); "
                           "0.0.0.0 exposes the unauthenticated dashboard to every "
                           "interface")
    # `server.headless` and "open a browser" are the same knob in Streamlit, so
    # these two are mutually exclusive rather than silently overriding each
    # other. Both default to None: "not given" means take [dashboard] headless.
    browser = dash.add_mutually_exclusive_group()
    browser.add_argument("--headless", dest="headless", action="store_true", default=None,
                         help="serve without opening a browser (default: [dashboard] headless)")
    browser.add_argument("--open", dest="headless", action="store_false",
                         help="open a browser once the server is listening")
    dash.set_defaults(func=_cmd_dashboard)

    # install
    install = add("install", help="install and enable the systemd user units")
    install.add_argument("--no-enable", action="store_true",
                         help="write the units but do not enable or start them")
    install.add_argument("--no-dashboard", action="store_true",
                         help="install only the collector, not the dashboard")
    install.set_defaults(func=_cmd_install)

    # uninstall
    uninstall = add("uninstall", help="stop, disable and remove the user units")
    uninstall.set_defaults(func=_cmd_uninstall)

    # auth
    auth = add("auth", help="manage dashboard accounts")
    auth_sub = auth.add_subparsers(dest="auth_action", metavar="ACTION", required=True)
    au_add = auth_sub.add_parser("add-user", help="create an account")
    au_add.add_argument("username", help="login name; case-insensitive")
    au_add.add_argument("--role", choices=("read", "write"), default="read",
                        help="read (default) or write")
    au_add.add_argument("--confirm", action="store_true",
                        help="required when creating the first account, so an "
                             "empty or mistyped run cannot leave the dashboard with "
                             "no way in")
    au_add.add_argument("--password", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    au_list = auth_sub.add_parser("list", help="list accounts")
    au_list.set_defaults(username=None, role=None, password=False)
    au_role = auth_sub.add_parser("set-role", help="change a role")
    au_role.add_argument("username")
    au_role.add_argument("--role", choices=("read", "write"), required=True)
    au_dis = auth_sub.add_parser("disable", help="revoke access without deleting")
    au_dis.add_argument("username")
    au_dis.set_defaults(role=None, password=False)
    au_rm = auth_sub.add_parser("remove", help="delete an account")
    au_rm.add_argument("username")
    au_rm.set_defaults(role=None, password=False)
    au_pw = auth_sub.add_parser("passwd", help="set a new password")
    au_pw.add_argument("username")
    au_pw.add_argument("--password", action="store_true", default=True,
                       help=argparse.SUPPRESS)
    au_secret = auth_sub.add_parser(
        "show-secret", help="report whether the proxy secret is set, without printing it"
    )
    au_secret.set_defaults(username=None, role=None, password=False)
    auth.set_defaults(func=_cmd_auth)

    # reset
    reset = add("reset", help="delete stored data")
    reset.add_argument("--data", action="store_true", help="delete stored observations")
    reset.add_argument("--logs", action="store_true", help="delete rotating log files")
    reset.set_defaults(func=_cmd_reset)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``est`` console script.

    Parameters
    ----------
    argv
        Argument list. Defaults to :data:`sys.argv`.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_global_defaults(args)

    if getattr(args, "command", None) is None:
        parser.print_help()
        return EXIT_ERROR

    try:
        config = load_config(
            args.config,
            use_user_config=not args.no_user_config,
        )
    except ConfigError as exc:
        _emit(_style()(str(exc), "red"))
        _emit(_style()("  run `est config init` to create a starter config", "dim"))
        return EXIT_ERROR

    if args.verbose:
        config.logging.level = "DEBUG"

    if not args.quiet:
        setup_logging(config)
    else:
        config.logging.level = "WARNING"
        setup_logging(config, quiet=True)

    try:
        return int(args.func(args, config))
    except TrackerError as exc:
        _emit(_style()(f"error: {exc}", "red"))
        return EXIT_ERROR
    except KeyboardInterrupt:
        _emit("\ninterrupted")
        return EXIT_SIGNAL
    finally:
        shutdown_logging()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
