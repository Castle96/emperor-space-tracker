"""Threshold rule engine with hysteresis, confirmation and cooldowns.

The problem this solves
-----------------------
A monitoring node polls every five minutes for months. Any threshold written the
obvious way -- ``if kp >= 5: alert()`` -- pages a human every single cycle for
the duration of a storm, which trains everyone to ignore the channel and makes
the *one* alert that matters invisible. A geomagnetic storm routinely holds Kp
at 6 or 7 for six to twelve hours: at a five-minute cadence that is 72-144
identical pages.

Four mechanisms prevent that, and they compose:

**Confirmation streak** (:attr:`RuleDefinition.confirm_samples`)
    A condition must hold for N consecutive polls before it alerts. Rejects
    single-sample noise from spacecraft faults and transient network glitches.
    A Kp spike lasting one minute is not a storm.

**Hysteresis** (``enter_threshold`` / ``clear_threshold``)
    Separate thresholds for entering and leaving a state. With ``storm_kp = 5.0``
    and ``storm_kp_clear = 4.0``, Kp oscillating across 4.5 produces one
    transition, not fifty. The gap is a config field precisely so the choice is
    explicit and reviewable.

**Cooldown** (``cooldown_seconds``)
    Even a genuine re-entry will not re-page within the cooldown window.

**Persistent latch** (in SQLite, not memory)
    Whether a condition is active survives a restart. A node that browns out
    mid-storm and comes back knowing it is already in a storm is the difference
    between one page and two; a node that forgets and re-pages on every boot is
    worse than no alerting at all.

State transitions are reported explicitly, so an operator can distinguish
"still storming" from "storm began" and "storm ended" -- the last of which is
the single most important message in the whole system, because it is the all
clear.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from itertools import pairwise
from typing import Any, Final

from ..config import SarConfig, SpaceWeatherConfig
from ..errors import StoreError
from ..models import (
    Colony,
    SarScene,
    Severity,
    SpaceWeatherSnapshot,
    utc_now,
)
from ..store import Store
from .notifier import Alert

__all__ = [
    "RuleContext",
    "RuleDefinition",
    "RuleEngine",
    "RuleOutcome",
    "RuleState",
    "build_default_rules",
]

_LOG = logging.getLogger("emperor.alerts.rules")

#: Kp values at or above which NOAA assigns a G-scale. Used in alert copy.
#: NOAA mapping: G1 = Kp 5, G2 = Kp 6, G3 = Kp 7, G4 = Kp 8, G5 = Kp 9.
G_SCALE: Final[dict[float, str]] = {
    5.0: "G1 minor storm",
    6.0: "G2 moderate storm",
    7.0: "G3 strong storm",
    8.0: "G4 severe storm",
    9.0: "G5 extreme storm",
}


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything the rules may inspect for one polling pass.

    Every field is optional because the sources fail independently, and a rule
    must be able to distinguish "the value is below threshold" from "the value
    could not be obtained". A rule that cannot tell those apart will eventually
    page someone about a sensor that is unplugged.
    """

    snapshot: SpaceWeatherSnapshot
    scenes: tuple[SarScene, ...] = ()
    colonies: tuple[Colony, ...] = ()
    now: datetime = field(default_factory=utc_now)
    stale_products: tuple[str, ...] = ()

    def fast_ice_scenes(self) -> list[SarScene]:
        """Return only the scenes that describe consolidated fast ice.

        The SAR client already declines to image anything that is not a
        fast-ice breeder, but a store can hold scenes written before a colony
        was reclassified, and the fast-ice rules must not act on a scene whose
        classification bands were never calibrated for its substrate. Filtering
        here makes the rule's correctness a property of the rule rather than a
        consequence of what the writer happened to do.
        """
        observed = {c.colony_id for c in self.colonies if c.monitors_fast_ice}
        if not observed:
            return []
        return [s for s in self.scenes if s.colony_id in observed]

    def latest_scene(self, colony_id: str) -> SarScene | None:
        """Return the newest scene for a colony, if any.

        Parameters
        ----------
        colony_id
            Colony identifier.

        Returns
        -------
        SarScene | None
        """
        candidates = [s for s in self.scenes if s.colony_id == colony_id]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.observed_at)

    def recent_series(self, colony_id: str, *, window: int) -> list[SarScene] | None:
        """Return a colony's most recent ``window`` scenes, or ``None``.

        ``None`` when there are too few scenes to say anything about a trend,
        which is different from a trend of zero and must not be collapsed into
        one.
        """
        series = sorted(
            (s for s in self.scenes if s.colony_id == colony_id),
            key=lambda s: s.observed_at,
        )
        if len(series) < window:
            return None
        return series[-window:]

    def trending_down(
        self,
        colony_id: str,
        *,
        window: int = 4,
        min_drop_db: float = 1.5,
    ) -> tuple[bool, float]:
        """Detect a sustained decline in mean sigma0 for a colony.

        A sustained fall in backscatter is the operational signature of a
        widening polynya: less consolidated ice, more open water, less volume
        scattering. A single low acquisition is a wind-roughening artefact, so
        the check requires a monotone trend across a window.

        This is the *change* detector, as distinct from the classification bands
        the SAR client applies to a single acquisition. The bands say what the
        ice looks like now; this says it is getting worse, which is the earlier
        and more actionable of the two signals.

        Parameters
        ----------
        colony_id
            Colony to inspect.
        window
            Number of scenes in the window.
        min_drop_db
            Total drop in dB across the window required to fire.

        Returns
        -------
        tuple[bool, float]
            ``(is_declining, total_drop_db)``. The drop is negative when sigma0
            is falling, matching the physical convention. An insufficient series
            returns ``(False, 0.0)``; use :meth:`trend_drop` when the difference
            between "flat" and "unknown" matters.
        """
        recent = self.recent_series(colony_id, window=window)
        if recent is None:
            return False, 0.0
        drop = recent[-1].mean_db - recent[0].mean_db
        if drop > -min_drop_db:
            return False, drop
        # Require it to be a trend, not a single outlier excursion.
        steps = [after.mean_db - before.mean_db for before, after in pairwise(recent)]
        declining = sum(1 for step in steps if step < 0)
        return declining >= max(1, (len(steps) + 1) // 2), drop

    def trend_drop(self, *, window: int = 4) -> float | None:
        """Return the steepest sustained sigma0 decline across fast-ice colonies.

        The clear-side counterpart to :meth:`trending_down`: a scalar for the
        hysteresis gate, so a declining trend releases once it recovers rather
        than only once the magnitude test stops holding.

        Returns
        -------
        float | None
            The most negative drop in dB, or ``None`` when no monitored colony
            has enough scenes to judge. ``None`` is not zero: it means the trend
            is undeterminable, and the caller must hold its latch rather than
            report an all clear on absent evidence.
        """
        drops: list[float] = []
        for colony in self.colonies:
            if not colony.monitors_fast_ice:
                continue
            recent = self.recent_series(colony.colony_id, window=window)
            if recent is None:
                continue
            drops.append(recent[-1].mean_db - recent[0].mean_db)
        if not drops:
            return None
        return min(drops)


@dataclass(slots=True)
class RuleState:
    """Mutable evaluation state for one rule, persisted across restarts.

    Attributes
    ----------
    latched
        Whether the condition is currently active.
    streak
        Consecutive passes the condition has held.
    last_evaluated
        When the rule last ran.
    """

    latched: bool = False
    streak: int = 0
    last_evaluated: datetime | None = None


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    """A single threshold rule.

    Parameters
    ----------
    rule_id
        Stable identifier, used as the cooldown and latch key.
    name
        Human label.
    severity
        Severity of the alert this rule raises.
    predicate
        Returns ``True`` when the *enter* condition holds.
    describe
        Renders the alert title, body and context from the current context.
        Called only when the rule actually fires.
    enter_threshold
        Documented threshold value, carried into the alert context.
    clear_threshold
        Documented hysteresis release value, if any.
    clear_value
        Extracts the current scalar value of the watched quantity from the
        context, or ``None`` when it is unavailable. When ``None`` (the
        default), the latch releases as soon as the enter predicate stops
        holding. When set, the release is governed by ``clear_threshold``
        instead, and an unavailable value *holds* the latch: missing data is
        not recovery.
    clear_below
        Release polarity for ``clear_value``. ``True`` (the default) releases
        when the value falls to or below ``clear_threshold`` (storm Kp, wind
        speed, ice fraction). ``False`` releases when the value rises to or
        above it (southward Bz recovering northward).
    confirm_samples
        Consecutive passes required before firing.
    cooldown_seconds
        Minimum seconds between two alerts from this rule.
    clear_description
        Optional body text for the all-clear message.

    Examples
    --------
    >>> from emperor_space_tracker.config import SpaceWeatherConfig
    >>> rule = build_default_rules(SpaceWeatherConfig())[0]
    >>> rule.rule_id
    'geomagnetic-storm'
    """

    rule_id: str
    name: str
    severity: Severity
    predicate: Callable[[RuleContext], bool]
    describe: Callable[[RuleContext], tuple[str, str, dict[str, Any]]]
    enter_threshold: float
    clear_threshold: float | None = None
    clear_value: Callable[[RuleContext], float | None] | None = None
    clear_below: bool = True
    confirm_samples: int = 1
    cooldown_seconds: float = 0.0
    clear_description: str | None = None


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    """What the engine decided about one rule this pass.

    Attributes
    ----------
    rule_id
        Rule identifier.
    state
        ``"fired"``, ``"cleared"``, ``"active"`` (already known), ``"pending"``
        (streak building) or ``"idle"``.
    alert
        The alert to dispatch, if any.
    suppressed
        ``True`` if an alert was warranted but cooldown or severity floor
        withheld it. Surfaced in the logs, because a silently dropped alert is
        indistinguishable from a working one.
    """

    rule_id: str
    state: str
    alert: Alert | None = None
    suppressed: bool = False
    detail: str = ""


class RuleEngine:
    """Evaluates rules against a context, with persistent suppression.

    Parameters
    ----------
    rules
        The rule set to evaluate.
    store
        Store holding rule latches and the alert history. May be ``None`` for
        a stateless engine, which is convenient in tests but will re-page on
        every restart.
    min_severity
        Alerts below this are evaluated and logged but not dispatched.
    now_fn
        Injectable clock, so cooldown behaviour is testable without sleeping.

    Examples
    --------
    >>> from emperor_space_tracker.config import SpaceWeatherConfig
    >>> rules = build_default_rules(SpaceWeatherConfig())
    >>> engine = RuleEngine(rules, store=None, min_severity=Severity.INFO)
    >>> snapshot = SpaceWeatherSnapshot(observed_at=utc_now())
    >>> outcomes = engine.evaluate(RuleContext(snapshot=snapshot))
    >>> {o.rule_id for o in outcomes} >= {"geomagnetic-storm"}
    True
    """

    def __init__(
        self,
        rules: Sequence[RuleDefinition],
        *,
        store: Store | None = None,
        min_severity: Severity = Severity.INFO,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.rules = list(rules)
        self.store = store
        self.min_severity = min_severity
        self.now_fn = now_fn
        self._state: dict[str, RuleState] = {r.rule_id: RuleState() for r in self.rules}
        self._restore()

    def _restore(self) -> None:
        """Reload latches and streaks from the store."""
        if self.store is None:
            return
        for rule in self.rules:
            try:
                persisted = self.store.get_rule_state(rule.rule_id)
            except StoreError as exc:
                _LOG.warning("cannot restore state for %s: %s", rule.rule_id, exc)
                continue
            if persisted is not None:
                latched, streak, _ = persisted
                self._state[rule.rule_id] = RuleState(latched=latched, streak=streak)

    def _persist(self, rule: RuleDefinition, state: RuleState, context: dict[str, Any]) -> None:
        """Write a rule's state back to the store."""
        if self.store is None:
            return
        try:
            self.store.set_rule_state(
                rule.rule_id, latched=state.latched, streak=state.streak, context=context
            )
        except StoreError as exc:
            _LOG.warning("cannot persist state for %s: %s", rule.rule_id, exc)

    def reset(self) -> None:
        """Clear all in-memory latches. Does not touch the store."""
        for state in self._state.values():
            state.latched = False
            state.streak = 0
            state.last_evaluated = None

    def evaluate(self, context: RuleContext) -> list[RuleOutcome]:
        """Evaluate every rule against one polling pass.

        Parameters
        ----------
        context
            The observations for this pass.

        Returns
        -------
        list[RuleOutcome]
            One outcome per rule, in rule order.
        """
        outcomes: list[RuleOutcome] = []
        for rule in self.rules:
            outcomes.append(self._evaluate_one(rule, context))
        return outcomes

    def _evaluate_one(self, rule: RuleDefinition, context: RuleContext) -> RuleOutcome:
        """Evaluate a single rule, handling streak, latch, cooldown and clear.

        Parameters
        ----------
        rule
            The rule to evaluate.
        context
            The observations for this pass.

        Returns
        -------
        RuleOutcome
        """
        state = self._state[rule.rule_id]
        now = context.now
        holds = self._safe_predicate(rule, context)
        state.last_evaluated = now

        # -- confirmation streak ------------------------------------------
        if holds:
            state.streak += 1
        else:
            state.streak = 0

        confirmed = holds and state.streak >= max(1, rule.confirm_samples)

        # -- entering ------------------------------------------------------
        if confirmed and not state.latched:
            state.latched = True
            self._persist(rule, state, {"entered_at": now.isoformat()})
            alert = self._build_alert(rule, context)
            if alert is None:
                return RuleOutcome(rule.rule_id, "suppressed", None, True, "below severity floor")
            if self._in_cooldown(rule, now):
                return RuleOutcome(
                    rule.rule_id,
                    "suppressed",
                    alert,
                    True,
                    f"within {rule.cooldown_seconds:g}s cooldown",
                )
            return RuleOutcome(rule.rule_id, "fired", alert, False)

        # -- clearing ------------------------------------------------------
        # A latch releases only once the *hysteresis* threshold is met, not
        # merely because the raw condition stopped holding. Without that
        # separation the latch would chatter exactly where hysteresis is meant
        # to help, so _should_clear is a distinct gate rather than a re-test.
        if not holds and state.latched and self._should_clear(rule, context):
            state.latched = False
            self._persist(rule, state, {"cleared_at": now.isoformat()})
            if rule.clear_description is not None:
                clear_alert = Alert(
                    rule_id=f"{rule.rule_id}-cleared",
                    severity=Severity.INFO,
                    title=f"Cleared: {rule.name}",
                    body=rule.clear_description,
                    context={"cleared_at": now.isoformat()},
                    fields=(
                        ("Node", context_colony_summary(context)),
                        ("Cleared at", now.isoformat(timespec="seconds")),
                    ),
                )
                return RuleOutcome(rule.rule_id, "cleared", clear_alert, False)
            return RuleOutcome(rule.rule_id, "cleared", None, False)

        if confirmed:
            self._persist(rule, state, {"active_since": now.isoformat()})
            return RuleOutcome(rule.rule_id, "active", None, False, "already latched")

        if holds:
            return RuleOutcome(
                rule.rule_id,
                "pending",
                None,
                False,
                f"streak {state.streak}/{rule.confirm_samples}",
            )
        return RuleOutcome(rule.rule_id, "idle")

    def _should_clear(self, rule: RuleDefinition, context: RuleContext) -> bool:
        """Return whether a latched rule may release.

        When a hysteresis threshold is configured, the release is governed by
        that value rather than by the enter predicate, which is the whole point
        of configuring it. The per-rule ``clear_value`` extractor supplies the
        current reading; an unavailable reading holds the latch, because
        releasing on a data gap would report an all clear that nobody has
        evidence for, which is the most dangerous message this system can send.

        Parameters
        ----------
        rule
            The rule being released.
        context
            Current observations.

        Returns
        -------
        bool
        """
        if rule.clear_threshold is None or rule.clear_value is None:
            return True
        value = rule.clear_value(context)
        if value is None:
            # No data is not "recovered". Hold the latch.
            return False
        if rule.clear_below:
            return value <= rule.clear_threshold
        return value >= rule.clear_threshold

    def _in_cooldown(self, rule: RuleDefinition, now: datetime) -> bool:
        """Return whether ``rule`` fired too recently to alert again.

        Parameters
        ----------
        rule
            The rule that wants to fire.
        now
            Current time.

        Returns
        -------
        bool
        """
        if self.store is None or rule.cooldown_seconds <= 0:
            return False
        try:
            last = self.store.last_alert_at(rule.rule_id)
        except StoreError:
            return False
        if last is None:
            return False
        return (now - last) < timedelta(seconds=rule.cooldown_seconds)

    def _build_alert(self, rule: RuleDefinition, context: RuleContext) -> Alert | None:
        """Render a rule's alert, or ``None`` if below the severity floor.

        Parameters
        ----------
        rule
            The firing rule.
        context
            Current observations.

        Returns
        -------
        Alert | None
        """
        if rule.severity.rank < self.min_severity.rank:
            return None
        title, body, values = rule.describe(context)
        fields = tuple((key, _render(value)) for key, value in values.items())
        return Alert(
            rule_id=rule.rule_id,
            severity=rule.severity,
            title=title,
            body=body,
            context=values,
            fields=fields,
        )

    @staticmethod
    def _safe_predicate(rule: RuleDefinition, context: RuleContext) -> bool:
        """Evaluate a rule's predicate, treating a fault as "not met".

        A rule raising on missing data must not take down the whole engine; it
        is logged and the condition is treated as unmet, which is the safe
        direction because a suppression is recoverable and a missed page is not.

        Parameters
        ----------
        rule
            The rule to evaluate.
        context
            Current observations.

        Returns
        -------
        bool
        """
        try:
            return bool(rule.predicate(context))
        except Exception as exc:
            _LOG.error("rule %s raised while evaluating: %s", rule.rule_id, exc, exc_info=True)
            return False

    def stale_warning(self, context: RuleContext) -> Alert | None:
        """Return an alert when any observation is too old to trust.

        .. deprecated::
            Superseded by the ``stale-observations`` rule in the default set,
            which carries a latch and a cooldown. This method re-raised the
            alert on every single pass -- at the default five-minute cadence,
            288 identical pages a day during a network partition, which is the
            exact failure the rule engine exists to prevent. It remains only so
            an out-of-tree caller keeps working; new code should rely on the
            rule's transition instead of its level.

        Parameters
        ----------
        context
            Current observations.

        Returns
        -------
        Alert | None
        """
        if not context.stale_products:
            return None
        return Alert(
            rule_id="stale-observations",
            severity=Severity.WARNING,
            title="Observation feed is stale",
            body=(
                "One or more upstream feeds are returning cached or stale data. "
                "Threshold rules are not being evaluated against current conditions, "
                "so a developing event may be missed. Check network egress and the "
                "upstream services."
            ),
            context={"stale": list(context.stale_products)},
            fields=tuple(
                (name.split(" is ")[0].replace("_", " ").title(), "stale")
                for name in context.stale_products
            ),
        )


def context_colony_summary(context: RuleContext) -> str:
    """Return a one-line summary of the colonies under watch.

    Parameters
    ----------
    context
        The current observations.

    Returns
    -------
    str
    """
    if not context.colonies:
        return "no colonies in range"
    major = [c for c in context.colonies if c.is_major]
    names = ", ".join(c.name for c in (major or context.colonies)[:3])
    suffix = f" +{len(context.colonies) - 3} more" if len(context.colonies) > 3 else ""
    return f"{names}{suffix}"


def _render(value: Any) -> str:
    """Render a context value for an alert field.

    Parameters
    ----------
    value
        The value.

    Returns
    -------
    str
    """
    if isinstance(value, float):
        return f"{value:.2f}"
    if value is None:
        return "unavailable"
    return str(value)


def _g_scale(kp: float) -> str:
    """Return the NOAA G-scale label for a Kp value.

    Parameters
    ----------
    kp
        The Kp index value.

    Returns
    -------
    str
    """
    for threshold in sorted(G_SCALE, reverse=True):
        if kp >= threshold:
            return G_SCALE[threshold]
    return "below storm threshold"


def build_default_rules(
    config: SpaceWeatherConfig,
    *,
    cooldown_seconds: float = 1800.0,
    sar: SarConfig | None = None,
) -> list[RuleDefinition]:
    """Build the default rule set from configuration.

    Six rules, covering the failure modes a fast-ice monitoring node actually
    needs to catch during polar night:

    1. **Geomagnetic storm** (Kp >= 5.0). Expands the auroral oval poleward;
       degrades HF radio and GNSS at high latitude, which is the colony's only
       link to the outside.
    2. **High-speed solar wind** (> 700 km/s). Compresses the magnetosphere;
       the leading indicator that usually precedes (1) by an hour.
    3. **Sustained southward Bz** (< -5 nT). The driver of main-phase
       reconnection, and the reason (1) happens.
    4. **Fast-ice retreat** at any colony. The project's primary mission, and
       an *absolute* judgement: what the ice looks like in the latest frame.
    5. **Polynya widening**: mean backscatter falling steadily across
       acquisitions. The *change* judgement, and the earlier of the two signals
       -- a colony whose ice is degrading is in trouble well before any single
       frame reads ``breached``.
    6. **Provenance / feed integrity**, handled separately by
       :meth:`RuleEngine.stale_warning`.

    Parameters
    ----------
    config
        The ``[space_weather]`` configuration section, supplying thresholds
        and confirmation counts.
    cooldown_seconds
        Minimum seconds between two alerts from the space-weather rules.
        Supplied by the caller from ``[alerts] cooldown_seconds``; the
        fast-ice rules keep their own longer cooldown.
    sar
        The ``[sar]`` section, supplying the polynya-trend window and
        thresholds. Defaults to :class:`SarConfig` defaults when omitted, so a
        caller that only cares about space weather still gets a complete set.

    Returns
    -------
    list[RuleDefinition]
        Rules in evaluation order.

    Examples
    --------
    >>> from emperor_space_tracker.config import load_config
    >>> cfg = load_config(use_user_config=False)
    >>> rules = build_default_rules(cfg.space_weather, sar=cfg.sar)
    >>> [r.rule_id for r in rules]  # doctest: +NORMALIZE_WHITESPACE
    ['geomagnetic-storm', 'high-speed-solar-wind', 'southward-bz',
     'fast-ice-retreat', 'polynya-widening', 'stale-observations']
    """
    sar_config = sar if sar is not None else SarConfig()
    polynya_enter, polynya_clear = sar_config.polynya_drop_band()
    polynya_window = max(2, sar_config.polynya_window_scenes)
    confirm = max(1, config.consecutive_samples)
    cooldown = cooldown_seconds

    def kp_value(ctx: RuleContext) -> float | None:
        """Return the current Kp, or ``None`` if unavailable."""
        return ctx.snapshot.kp.value if ctx.snapshot.kp else None

    def storm(ctx: RuleContext) -> bool:
        """Return whether Kp is at or above the storm threshold."""
        value = kp_value(ctx)
        return value is not None and value >= config.storm_kp

    def fast_wind(ctx: RuleContext) -> bool:
        """Return whether the solar wind exceeds the high-speed threshold."""
        plasma = ctx.snapshot.plasma
        return (
            plasma is not None
            and plasma.speed_kms is not None
            and plasma.is_nominal()
            and plasma.speed_kms >= config.high_wind_kms
        )

    def southward(ctx: RuleContext) -> bool:
        """Return whether Bz GSM is sufficiently southward."""
        field_data = ctx.snapshot.magnetic_field
        return (
            field_data is not None
            and field_data.bz_gsm_nt is not None
            and field_data.is_nominal()
            and field_data.bz_gsm_nt <= config.southward_bz_nt
        )

    def ice_retreat(ctx: RuleContext) -> bool:
        """Return whether any fast-ice colony's platform has breached its band."""
        return any(
            scene.classify() in {"breached", "dispersed"} for scene in ctx.fast_ice_scenes()
        )

    def wind_speed(ctx: RuleContext) -> float | None:
        """Return the current solar wind speed, or ``None`` if unavailable."""
        plasma = ctx.snapshot.plasma
        return plasma.speed_kms if plasma is not None else None

    def bz_value(ctx: RuleContext) -> float | None:
        """Return the current IMF Bz (GSM), or ``None`` if unavailable."""
        field_data = ctx.snapshot.magnetic_field
        return field_data.bz_gsm_nt if field_data is not None else None

    def ice_water(ctx: RuleContext) -> float | None:
        """Return the worst open-water fraction across fast-ice scenes.

        ``None`` when there are no fast-ice scenes, which holds the latch
        rather than reporting an all clear with no evidence.
        """
        scenes = ctx.fast_ice_scenes()
        if not scenes:
            return None
        return max(scene.open_water_fraction for scene in scenes)

    def polynya_widening(ctx: RuleContext) -> bool:
        """Return whether any fast-ice colony's backscatter is trending down.

        Delegates to :meth:`RuleContext.trending_down` so the magnitude and
        monotonicity tests stay in one place, and restricts the search to
        fast-ice breeders because the trend bands are only meaningful over
        consolidated ice.
        """
        return any(
            ctx.trending_down(
                colony.colony_id,
                window=polynya_window,
                min_drop_db=abs(polynya_enter),
            )[0]
            for colony in ctx.colonies
            if colony.monitors_fast_ice
        )

    def polynya_drop(ctx: RuleContext) -> float | None:
        """Return the steepest sustained decline across fast-ice colonies.

        ``None`` when no colony has enough scenes to judge, which holds the
        latch: an undeterminable trend is not a recovered one.
        """
        return ctx.trend_drop(window=polynya_window)

    def feed_stale(ctx: RuleContext) -> bool:
        """Return whether any upstream product is older than it should be.

        The anti-silence guarantee. A node whose feeds are dead but which
        reports nothing is indistinguishable from a node with genuinely quiet
        space weather, and during polar night nobody notices the difference for
        weeks.
        """
        return bool(ctx.stale_products)

    def stale_count(ctx: RuleContext) -> float:
        """Return how many products are stale, as the hysteresis scalar.

        The count is zero exactly when the feeds are healthy, so it releases the
        latch on recovery with no separate scalar to keep in step.
        """
        return float(len(ctx.stale_products))

    storm_rule = RuleDefinition(
        rule_id="geomagnetic-storm",
        name="Geomagnetic storm",
        severity=Severity.SEVERE,
        predicate=storm,
        enter_threshold=config.storm_kp,
        clear_threshold=config.storm_kp_clear,
        clear_value=kp_value,
        clear_below=True,
        confirm_samples=confirm,
        cooldown_seconds=cooldown,
        clear_description=(
            f"Kp has fallen to or below {config.storm_kp_clear:g}. The auroral oval has "
            "retracted and high-latitude HF propagation and GNSS accuracy are recovering."
        ),
        describe=lambda ctx: _describe_storm(ctx, config),
    )

    wind_rule = RuleDefinition(
        rule_id="high-speed-solar-wind",
        name="High-speed solar wind stream",
        severity=Severity.WARNING,
        predicate=fast_wind,
        enter_threshold=config.high_wind_kms,
        clear_threshold=config.high_wind_kms * 0.9,
        clear_value=wind_speed,
        clear_below=True,
        confirm_samples=confirm,
        cooldown_seconds=cooldown,
        clear_description=(
            f"Solar wind speed has fallen below {config.high_wind_kms * 0.9:.0f} km/s. "
            "Magnetospheric dynamic pressure is returning to ambient levels."
        ),
        describe=lambda ctx: _describe_wind(ctx, config),
    )

    bz_rule = RuleDefinition(
        rule_id="southward-bz",
        name="Sustained southward IMF Bz",
        severity=Severity.WARNING,
        predicate=southward,
        enter_threshold=config.southward_bz_nt,
        clear_threshold=config.southward_bz_nt * 0.5,
        clear_value=bz_value,
        clear_below=False,
        confirm_samples=confirm,
        cooldown_seconds=cooldown,
        clear_description=(
            "The interplanetary magnetic field has rotated northward. Main-phase "
            "reconnection has ceased and the ring current is decaying."
        ),
        describe=lambda ctx: _describe_bz(ctx, config),
    )

    ice_rule = RuleDefinition(
        rule_id="fast-ice-retreat",
        name="Fast-ice retreat detected",
        severity=Severity.CRITICAL,
        predicate=ice_retreat,
        enter_threshold=0.15,
        clear_threshold=0.05,
        clear_value=ice_water,
        clear_below=True,
        confirm_samples=1,
        cooldown_seconds=3600.0,
        clear_description=(
            "Fast-ice conditions have returned to within tolerance at all monitored "
            "colonies. The breeding platform appears to be intact."
        ),
        describe=_describe_ice,
    )

    polynya_rule = RuleDefinition(
        rule_id="polynya-widening",
        name="Polynya widening",
        severity=Severity.SEVERE,
        predicate=polynya_widening,
        enter_threshold=polynya_enter,
        clear_threshold=polynya_clear,
        clear_value=polynya_drop,
        # The metric here is a signed dB *change*, not a magnitude: falling ice
        # gives a negative drop, so "recovered" is a *higher* value. Clearing is
        # therefore `drop >= clear`, i.e. `clear_below=False`. Inverting this
        # releases the latch while the ice is still degrading, which is the
        # exact opposite of what hysteresis is for.
        clear_below=False,
        # The trend test already averages over a window of acquisitions, so
        # requiring further consecutive confirmations would only delay a signal
        # that is inherently slow to build.
        confirm_samples=1,
        cooldown_seconds=cooldown,
        clear_description=(
            "Mean backscatter has recovered across the trend window at all "
            "monitored colonies. The fast-ice platform appears to be "
            "re-consolidating."
        ),
        describe=partial(_describe_polynya, window=polynya_window),
    )

    stale_rule = RuleDefinition(
        rule_id="stale-observations",
        name="Observation feed is stale",
        severity=Severity.WARNING,
        predicate=feed_stale,
        enter_threshold=1.0,
        # Zero stale products releases the latch. `clear_below` is True because
        # a *lower* count is the healthy state, unlike the signed-dB trend rule
        # where recovery is a higher value.
        clear_threshold=0.0,
        clear_value=stale_count,
        clear_below=True,
        confirm_samples=1,
        cooldown_seconds=cooldown,
        clear_description=(
            "All upstream observation feeds are reporting current data again. "
            "Threshold rules are being evaluated against live conditions."
        ),
        describe=_describe_stale,
    )

    return [storm_rule, wind_rule, bz_rule, ice_rule, polynya_rule, stale_rule]


def _describe_storm(
    ctx: RuleContext,
    config: SpaceWeatherConfig,
) -> tuple[str, str, dict[str, Any]]:
    """Render the geomagnetic storm alert.

    Parameters
    ----------
    ctx
        Current observations.
    config
        Space weather configuration.

    Returns
    -------
    tuple[str, str, dict[str, Any]]
        ``(title, body, context)``.
    """
    kp = kp_value_or_zero(ctx)
    scale = _g_scale(kp)
    return (
        f"Geomagnetic storm in progress: Kp {kp:.1f} ({scale})",
        (
            f"Planetary Kp has reached {kp:.1f}, at or above the alerting threshold of "
            f"{config.storm_kp:g}. At this level the auroral oval expands far enough "
            "south to degrade HF radio propagation and cause GNSS scintillation over "
            "Antarctica. Any field party on the ice should expect loss of HF contact "
            "and degraded satellite positioning, and should confirm its schedule with "
            f"the regional forecast centre. The all clear is Kp <= {config.storm_kp_clear:g}."
        ),
        {
            "kp": kp,
            "g_scale": scale,
            "threshold": config.storm_kp,
            "clear_at": config.storm_kp_clear,
            "observed_at": ctx.snapshot.observed_at.isoformat(timespec="seconds"),
            "colonies": context_colony_summary(ctx),
            "is_nowcast": ctx.snapshot.kp.is_estimated if ctx.snapshot.kp else None,
        },
    )


def _describe_wind(ctx: RuleContext, config: SpaceWeatherConfig) -> tuple[str, str, dict[str, Any]]:
    """Render the high-speed solar wind alert.

    Parameters
    ----------
    ctx
        Current observations.
    config
        Space weather configuration.

    Returns
    -------
    tuple[str, str, dict[str, Any]]
    """
    plasma = ctx.snapshot.plasma
    speed = plasma.speed_kms if plasma and plasma.speed_kms is not None else 0.0
    return (
        f"High-speed solar wind: {speed:.0f} km/s",
        (
            f"Bulk solar wind speed has reached {speed:.0f} km/s, above the "
            f"{config.high_wind_kms:g} km/s alerting threshold. Dynamic pressure on the "
            "magnetosphere is elevated, which compresses the magnetotail and typically "
            f"precedes a geomagnetic storm by one to three hours. Expect Kp to rise; "
            "the storm alert will follow if it does."
        ),
        {
            "speed_kms": speed,
            "threshold_kms": config.high_wind_kms,
            "density_per_cm3": plasma.density_per_cm3 if plasma else None,
            "spacecraft": plasma.source if plasma else None,
            "observed_at": ctx.snapshot.observed_at.isoformat(timespec="seconds"),
        },
    )


def _describe_bz(ctx: RuleContext, config: SpaceWeatherConfig) -> tuple[str, str, dict[str, Any]]:
    """Render the southward Bz alert.

    Parameters
    ----------
    ctx
        Current observations.
    config
        Space weather configuration.

    Returns
    -------
    tuple[str, str, dict[str, Any]]
    """
    field_data = ctx.snapshot.magnetic_field
    bz = field_data.bz_gsm_nt if field_data and field_data.bz_gsm_nt is not None else 0.0
    return (
        f"Sustained southward Bz: {bz:+.1f} nT",
        (
            f"The GSM southward component of the interplanetary magnetic field is "
            f"{bz:+.1f} nT, at or beyond the {config.southward_bz_nt:+.1f} nT threshold. "
            "Southward Bz is the coupling mechanism that drives main-phase reconnection "
            "and the main phase of a geomagnetic storm. HF absorption in the polar "
            "caps is elevated while this persists."
        ),
        {
            "bz_gsm_nt": bz,
            "bt_nt": field_data.bt_nt if field_data else None,
            "threshold_nt": config.southward_bz_nt,
            "spacecraft": field_data.source if field_data else None,
            "observed_at": ctx.snapshot.observed_at.isoformat(timespec="seconds"),
        },
    )


def _describe_ice(ctx: RuleContext) -> tuple[str, str, dict[str, Any]]:
    """Render the fast-ice retreat alert.

    Parameters
    ----------
    ctx
        Current observations.

    Returns
    -------
    tuple[str, str, dict[str, Any]]
    """
    breached = [s for s in ctx.fast_ice_scenes() if s.classify() in {"breached", "dispersed"}]
    if not breached:
        # Unreachable while the predicate guards on the same set, but a
        # `max()` on an empty sequence would raise inside alert rendering,
        # which is the worst possible time for an exception.
        return (
            "Fast-ice retreat detected",
            "A fast-ice stability band was breached but the scene could not be "
            "resolved to a colony.",
            {},
        )
    worst = max(breached, key=lambda s: s.open_water_fraction)
    colony = next((c for c in ctx.colonies if c.colony_id == worst.colony_id), None)
    colony_name = colony.name if colony else worst.colony_id
    population = colony.population_estimate if colony else None
    source = "sentinel-1 SAR" if not worst.is_synthetic else "simulated SAR"
    band = worst.polarisation
    species = colony.species if colony else "the colony's species"
    common = colony.common_name if colony else ""
    subject = f"{common} ({species})" if common else species
    return (
        f"Fast-ice retreat at {colony_name} ({worst.classify()})",
        (
            f"{worst.open_water_fraction_percent:.1f}% of the analysis window around "
            f"{colony_name} is classified as open water, and mean C-band {band} "
            f"backscatter has fallen to {worst.mean_db:.1f} dB. The fast-ice platform "
            f"the colony depends on for its breeding cycle is no longer continuous"
            + (
                f", and the colony is estimated at {population:,} breeding pairs."
                if population
                else "."
            )
            + f" {subject} cannot complete a breeding cycle without attached, "
            "consolidated fast ice; if this persists through the incubation period the "
            "colony's reproductive success for the season is effectively lost. "
            "Confirm against a second acquisition before acting, and prioritise aerial "
            "reconnaissance if the platform is load-bearing for access."
        ),
        {
            "colony": colony_name,
            "population_estimate": population,
            "stability": worst.classify(),
            "open_water_pct": worst.open_water_fraction_percent,
            "mean_db": worst.mean_db,
            "min_db": worst.min_db,
            "max_db": worst.max_db,
            "std_db": worst.std_db,
            "scene_id": worst.scene_id,
            "observed_at": worst.observed_at.isoformat(timespec="seconds"),
            "data_source": source,
            "synthetic": worst.is_synthetic,
        },
    )


def _describe_polynya(ctx: RuleContext, *, window: int) -> tuple[str, str, dict[str, Any]]:
    """Render the polynya-widening alert.

    Reports the whole trend window rather than a single frame, because the
    judgement is about the *slope* between acquisitions and an operator reading
    only the newest sigma0 has no way to see that.

    Parameters
    ----------
    ctx
        Current observations.
    window
        The same window the predicate used. Bound by the caller with
        :func:`functools.partial` so the message can never quote a different
        window than the decision was made on.

    Returns
    -------
    tuple[str, str, dict[str, Any]]
        ``(title, body, context)``.
    """
    candidates: list[tuple[float, Colony, list[SarScene]]] = []
    for colony in ctx.colonies:
        if not colony.monitors_fast_ice:
            continue
        series = ctx.recent_series(colony.colony_id, window=window)
        if series is None:
            continue
        candidates.append((series[-1].mean_db - series[0].mean_db, colony, series))

    if not candidates:
        return (
            "Polynya widening detected",
            "A sustained backscatter decline was detected but no colony could be "
            "resolved to a trend window.",
            {},
        )

    drop, colony, series = min(candidates, key=lambda item: item[0])
    series_text = " -> ".join(f"{s.mean_db:.2f}" for s in series)
    times_text = " -> ".join(s.observed_at.strftime("%d %b %H:%M") for s in series)
    synthetic = all(s.is_synthetic for s in series)
    return (
        f"Polynya widening at {colony.name} ({drop:+.2f} dB)",
        (
            f"Mean backscatter over the analysis window around {colony.name} has "
            f"fallen {abs(drop):.2f} dB across the last {len(series)} acquisitions "
            f"({times_text}): {series_text} dB. A sustained decline like this is the "
            "signature of a polynya opening or the fast ice detaching from the "
            "coast: less consolidated ice, more open water, less volume scattering. "
            "It is the earlier warning -- the absolute-band alert only fires once a "
            "single frame already reads breached. Confirm against the next "
            "acquisition before committing to aerial reconnaissance; a single "
            "acquisition can also fall from wind roughening, which is why the "
            "detector requires the decline to hold across the whole window."
        ),
        {
            "colony": colony.name,
            "species": colony.species,
            "drop_db": round(drop, 3),
            "window_scenes": len(series),
            "mean_db_series": [round(s.mean_db, 3) for s in series],
            "open_water_pct_series": [
                round(s.open_water_fraction_percent, 2) for s in series
            ],
            "first_observed_at": series[0].observed_at.isoformat(timespec="seconds"),
            "last_observed_at": series[-1].observed_at.isoformat(timespec="seconds"),
            "data_source": "simulated SAR" if synthetic else "sentinel-1 SAR",
            "synthetic": synthetic,
        },
    )


def _describe_stale(ctx: RuleContext) -> tuple[str, str, dict[str, Any]]:
    """Render the stale-feed alert.

    Names the specific products that are stale rather than saying "something is
    stale", because the operator's next action differs per product: a stale
    space-weather feed is a caching proxy, a stale SAR feed is usually the
    constellation's orbit geometry.

    Parameters
    ----------
    ctx
        Current observations.

    Returns
    -------
    tuple[str, str, dict[str, Any]]
        ``(title, body, context)``.
    """
    stale = list(ctx.stale_products)
    named = ", ".join(stale)
    return (
        f"Observation feed is stale ({len(stale)} product(s))",
        (
            f"The following upstream products are returning cached or stale data: "
            f"{named}. Threshold rules are not being evaluated against current "
            "conditions, so a developing storm or lead event may be missed. The node "
            "is alive and reporting; what it is reporting is not current. Check "
            "network egress to the upstream services first, and note that a missing "
            "orbit pass will also look like this."
        ),
        {
            "stale_products": stale,
            "stale_count": len(stale),
            "observed_at": ctx.now.isoformat(timespec="seconds"),
        },
    )


def kp_value_or_zero(ctx: RuleContext) -> float:
    """Return the current Kp, or ``0.0`` when unavailable.

    Parameters
    ----------
    ctx
        Current observations.

    Returns
    -------
    float
    """
    value = ctx.snapshot.kp.value if ctx.snapshot.kp else None
    return value if value is not None else 0.0
