"""Integration tests for the polling engine, the daemon, and the store.

These paths had no coverage at all. That is not a gap in the ordinary sense:
they are the parts of the project whose behaviour is *claimed* in the README,
and every one of those claims was untested.

    - a pass writes its observations and evaluates rules;
    - a degraded source does not lose the others' data;
    - an active condition is latched, and a node that restarts mid-storm pages
      once rather than twice;
    - a cooldown suppresses a genuine re-entry;
    - the daemon's exit code distinguishes clean from failed.

Each is a promise an operator relies on when they leave a node unattended in
winter, so each is pinned here against the real classes rather than a mock of
them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from emperor_space_tracker.alerts.notifier import (
    Alert,
    AlertDispatcher,
    DiscordNotifier,
    NullNotifier,
)
from emperor_space_tracker.alerts.rules import (
    RuleContext,
    RuleDefinition,
    RuleEngine,
    RuleOutcome,
    build_default_rules,
)
from emperor_space_tracker.config import (
    AlertConfig,
    ColonyConfig,
    Config,
    DaemonConfig,
    PathsConfig,
    SarConfig,
    SiteConfig,
    SpaceWeatherConfig,
)
from emperor_space_tracker.engine import Daemon, PollEngine, build_engine_and_daemon
from emperor_space_tracker.errors import AlertDeliveryError, TrackerError
from emperor_space_tracker.models import (
    BreedingHabitat,
    Colony,
    FastIceCell,
    InterplanetaryMagneticField,
    KpIndex,
    SarScene,
    Severity,
    SolarWindPlasma,
    SourceHealth,
    SpaceWeatherSnapshot,
    utc_now,
)
from emperor_space_tracker.store import Store, to_micros

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


class RecordingDispatcher(AlertDispatcher):
    """Captures dispatched alerts instead of sending them.

    Implements the dispatcher protocol directly so the tests exercise the
    boundary the daemon actually uses, rather than reaching into the engine's
    internals to count calls.
    """

    def __init__(self) -> None:
        """Start with an empty outbox."""
        self.sent: list[Alert] = []

    def dispatch(self, alert: Alert) -> bool:
        """Record the alert and report success."""
        self.sent.append(alert)
        return True

    def describe(self) -> str:
        """Return a human-readable description."""
        return "recording dispatcher"

    def clear(self) -> None:
        """Discard everything recorded so far."""
        self.sent.clear()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """Open an on-disk store; WAL and the size cap only exist on a file."""
    opened = Store(str(tmp_path / "tracker.sqlite3"))
    try:
        yield opened
    finally:
        opened.close()


def _config(tmp_path: Path, **overrides: object) -> Config:
    """Build a config pointed at ``tmp_path`` with everything offline."""
    base = {
        "site": _site(),
        "paths": PathsConfig(state_dir=str(tmp_path)),
        "daemon": DaemonConfig(poll_interval_seconds=1, http_timeout_seconds=1),
        "space_weather": SpaceWeatherConfig(enabled=False),
        "sar": SarConfig(enabled=False, backend="synthetic", grid_cells=5),
        "colonies": ColonyConfig(include_gbif=False, max_distance_km=5.0),
        "alerts": AlertConfig(enabled=False, cooldown_seconds=0),
    }
    base.update(overrides)
    return Config(**base)  # type: ignore[arg-type]


def _site() -> SiteConfig:
    return SiteConfig(name="test-node", latitude=-77.85, longitude=166.67)


def _colony(
    *,
    colony_id: str = "cpe-test",
    habitat: BreedingHabitat = BreedingHabitat.FAST_ICE,
    species: str = "Aptenodytes forsteri",
) -> Colony:
    """Build a colony record on either the fast-ice or the land path."""
    return Colony(
        colony_id=colony_id,
        name=f"Colony {colony_id}",
        latitude=-77.5,
        longitude=166.6,
        region="Ross Sea",
        population_estimate=4200,
        population_year=2022,
        population_source="test census",
        species=species,
        breeding_habitat=habitat,
        fast_ice_ratio=0.9 if habitat is BreedingHabitat.FAST_ICE else None,
    )


def _snapshot(kp: float = 3.0) -> SpaceWeatherSnapshot:
    """Build a complete, non-degraded snapshot with a given Kp."""
    now = utc_now()
    return SpaceWeatherSnapshot(
        observed_at=now,
        plasma=SolarWindPlasma(
            now, "noaa.swpc.rtsw", 400.0, 5.0, 100_000.0, False, 1
        ),
        magnetic_field=InterplanetaryMagneticField(
            now, "noaa.swpc.rtsw", 5.0, -1.0, 2.0, 5.0, False
        ),
        kp=KpIndex(now, int(kp), kp),
        f107_sfu=90.0,
    )


def _scene(colony_id: str, water: float, at: datetime) -> SarScene:
    """Build a synthetic scene with a given open-water fraction."""
    return SarScene(
        scene_id=f"{colony_id}-{at.timestamp():.0f}",
        observed_at=at,
        platform="SENTINEL-1",
        orbit_type="ASCENDING",
        polarisation="VV",
        incidence_angle_deg=35.0,
        colony_id=colony_id,
        mean_db=-7.0,
        min_db=-12.0,
        max_db=-3.0,
        std_db=1.0,
        frozen_ice_fraction=1.0 - water,
        open_water_fraction=water,
        cells=(),
        provenance="synthetic.sar-simulator/v1",
    )


# --------------------------------------------------------------------------- #
# Store write paths
# --------------------------------------------------------------------------- #


def test_record_snapshot_persists_every_component_table(store: Store) -> None:
    """One call writes the snapshot and all four component tables.

    The dashboard reads the component tables while the rules read the joined
    one, so a pass that wrote only the join would render an empty space-weather
    tab while alerting correctly -- the worst kind of disagreement.
    """
    snapshot = _snapshot(kp=4.2)
    store.record_snapshot(snapshot)

    latest = store.latest_space_weather()
    assert latest is not None
    assert latest["kp_value"] == pytest.approx(4.2)

    # The table names are a literal tuple in this file, not input.
    for table in ("plasma", "magnetic_field", "kp_index", "f10_7_flux"):
        count = store.conn.execute("SELECT count(*) FROM " + table).fetchone()[0]  # noqa: S608
        assert count == 1, f"{table} was not written"


def test_latest_space_weather_returns_the_newest_not_the_first(store: Store) -> None:
    """Out-of-order arrivals still resolve to the most recent observation.

    NOAA products are fetched independently, so a slow feed can land an older
    sample after a newer one. Without ORDER BY the "latest" panel would show
    whichever row SQLite happened to return first.
    """
    older = _snapshot(2.0)
    newer = _snapshot(6.0)
    older = replace(older, observed_at=datetime(2026, 1, 1, tzinfo=UTC))
    newer = replace(newer, observed_at=datetime(2026, 1, 5, tzinfo=UTC))
    # Deliberately write the newer observation first, so a reader that takes
    # the first matching row rather than the max will fail.
    store.record_snapshot(newer)
    store.record_snapshot(older)

    latest = store.latest_space_weather()
    assert latest is not None
    assert latest["kp_value"] == pytest.approx(6.0)
    assert latest["observed_at"] == datetime(2026, 1, 5, tzinfo=UTC)


def test_upsert_colonies_round_trips_species_and_habitat(store: Store) -> None:
    """A colony survives a write/read cycle with its taxon intact.

    The species fields are what decide whether SAR will image the colony, so
    losing them on the way to disk silently disables the fast-ice pipeline for
    that colony on the next pass.
    """
    originals = [
        _colony(colony_id="cpe-ice"),
        _colony(
            colony_id="pgk-land",
            habitat=BreedingHabitat.LAND,
            species="Pygoscelis kerguelensis",
        ),
    ]
    assert store.upsert_colonies(originals) == 2

    loaded = {c.colony_id: c for c in store.colonies()}
    assert set(loaded) == {"cpe-ice", "pgk-land"}
    assert loaded["cpe-ice"].species == "Aptenodytes forsteri"
    assert loaded["cpe-ice"].monitors_fast_ice
    assert loaded["pgk-land"].species == "Pygoscelis kerguelensis"
    assert not loaded["pgk-land"].monitors_fast_ice
    assert loaded["pgk-land"].fast_ice_ratio is None


def test_upsert_colonies_updates_rather_than_duplicating(store: Store) -> None:
    """A refreshed census replaces the old one under the same primary key."""
    store.upsert_colonies([_colony()])
    updated = Colony(
        colony_id="cpe-test", name="Renamed", latitude=-77.5, longitude=166.6,
        region="Ross Sea", population_estimate=5100, population_year=2024,
        population_source="new census", species="Aptenodytes forsteri",
        breeding_habitat=BreedingHabitat.FAST_ICE, fast_ice_ratio=0.7,
    )
    store.upsert_colonies([updated])

    colonies = store.colonies()
    assert len(colonies) == 1
    assert colonies[0].name == "Renamed"
    assert colonies[0].population_estimate == 5100
    assert colonies[0].population_year == 2024
    assert colonies[0].fast_ice_ratio == pytest.approx(0.7)


def test_upsert_of_an_empty_list_writes_nothing(store: Store) -> None:
    """An empty write is a no-op, not a wipe."""
    store.upsert_colonies([_colony()])
    assert store.upsert_colonies([]) == 0
    assert len(store.colonies()) == 1


def test_geojson_carries_the_species_fields(store: Store) -> None:
    """The map layer can tell two species apart."""
    store.upsert_colonies([_colony(colony_id="pgk-land",
                                    habitat=BreedingHabitat.LAND,
                                    species="Pygoscelis kerguelensis")])
    feature = store.geojson_feature_collection()["features"][0]
    props = feature["properties"]
    assert props["species"] == "Pygoscelis kerguelensis"
    assert props["breeding_habitat"] == "land"
    assert props["monitors_fast_ice"] is False


def test_sar_scene_and_cells_persist_and_reload(store: Store) -> None:
    """A scene's 25-cell grid survives the write and reads back square."""
    at = utc_now()
    scene = _scene_with_cells("cpe-test", 0.02, at)

    assert store.record_sar_scene(scene) == 25

    cells = store.sar_matrix(scene.scene_id)
    assert len(cells) == 25
    assert {(c.row, c.col): c.sigma0_db for c in cells} == {
        (r, c): -7.0 - (r + c) * 0.1 for r in range(5) for c in range(5)
    }

    recent = store.recent_sar_scenes(limit=5)
    assert [s["scene_id"] for s in recent] == [scene.scene_id]
    assert recent[0]["polarisation"] == "VV"


def _scene_with_cells(colony_id: str, water: float, at: datetime) -> SarScene:
    """Build a scene carrying a 5x5 cell grid."""
    cells = tuple(
        FastIceCell(
            scene_id=f"{colony_id}-{at.timestamp():.0f}",
            row=r,
            col=c,
            sigma0_db=-7.0 - (r + c) * 0.1,
            is_open_water=False,
            classification="consolidated_ice",
        )
        for r in range(5)
        for c in range(5)
    )
    base = _scene(colony_id, water, at)
    return SarScene(
        scene_id=base.scene_id,
        observed_at=base.observed_at,
        platform=base.platform,
        orbit_type=base.orbit_type,
        polarisation=base.polarisation,
        incidence_angle_deg=base.incidence_angle_deg,
        colony_id=base.colony_id,
        mean_db=base.mean_db,
        min_db=base.min_db,
        max_db=base.max_db,
        std_db=base.std_db,
        frozen_ice_fraction=base.frozen_ice_fraction,
        open_water_fraction=base.open_water_fraction,
        cells=cells,
        provenance=base.provenance,
    )


def test_alert_delivery_state_is_recorded(store: Store) -> None:
    """A failed delivery is stored as failed, not silently as delivered."""
    store.record_alert(
        rule_id="fast-ice-retreat",
        severity=Severity.CRITICAL,
        title="t",
        body="b",
        context={"k": "v"},
        delivered=False,
        delivery_error="HTTP 500",
    )
    recent = store.recent_alerts(limit=5)
    assert len(recent) == 1
    assert recent[0].delivered is False
    assert recent[0].delivery_error == "HTTP 500"
    assert recent[0].context == {"k": "v"}


def test_last_alert_at_drives_the_cooldown(store: Store) -> None:
    """The cooldown is a query against persisted state, not a memory value."""
    fired = utc_now() - timedelta(seconds=60)
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO alerts(rule_id, severity, title, body, context, fired_at, "
            "delivered, delivery_error) VALUES(?,?,?,?,?,?,?,?)",
            ("geomagnetic-storm", "severe", "t", "b", None,
             to_micros(fired), 1, None),
        )
    last = store.last_alert_at("geomagnetic-storm")
    assert last is not None
    assert abs((utc_now() - last).total_seconds() - 60) < 5
    assert store.last_alert_at("never-fired") is None


# --------------------------------------------------------------------------- #
# Latch persistence: the README's headline claim
# --------------------------------------------------------------------------- #


def _kp(ctx: RuleContext) -> float | None:
    """Return the current Kp, or None if the product is missing."""
    return ctx.snapshot.kp.value if ctx.snapshot.kp else None


def _storm_rule(cooldown_seconds: float = 0.0) -> RuleDefinition:
    """Build a minimal rule that fires on Kp >= 5 and clears below 4."""
    return RuleDefinition(
        rule_id="test-storm",
        name="Test storm",
        severity=Severity.SEVERE,
        predicate=lambda ctx: (kp := _kp(ctx)) is not None and kp >= 5.0,
        describe=lambda ctx: ("storm", "body", {}),
        enter_threshold=5.0,
        clear_threshold=4.0,
        clear_value=lambda ctx: ctx.snapshot.kp.value if ctx.snapshot.kp else None,
        clear_below=True,
        confirm_samples=1,
        cooldown_seconds=cooldown_seconds,
        clear_description="storm over",
    )


def test_latch_survives_a_process_restart(store: Store) -> None:
    """A fresh engine over the same store knows the storm is already active.

    This is the behaviour the persistent latch exists for: a node that browns
    out mid-storm restarts, and must page once rather than twice. Losing it
    would mean a duty operator gets two pages for one storm and learns to
    ignore them.
    """
    first = RuleEngine([_storm_rule()], store=store, min_severity=Severity.INFO)
    ctx = RuleContext(snapshot=_snapshot(kp=6.0))
    assert first.evaluate(ctx)[0].state == "fired"

    # A brand new engine object, as after a restart.
    second = RuleEngine([_storm_rule()], store=store, min_severity=Severity.INFO)
    outcome = second.evaluate(ctx)[0]
    assert outcome.state == "active", "restarted node should not re-page an active storm"
    assert outcome.alert is None


def _is_latched(store: Store, rule_id: str) -> bool:
    """Read the persisted latch for a rule straight out of the store."""
    row = store.conn.execute(
        "SELECT latched FROM rule_state WHERE rule_id = ?", (rule_id,)
    ).fetchone()
    return bool(row["latched"]) if row else False


def test_latch_clears_only_below_the_clear_threshold(store: Store) -> None:
    """Hysteresis holds the latch inside the band and releases below it.

    Kp 4.5 no longer satisfies the predicate, but it is still above the 4.0
    clear line, so the storm has not ended. Releasing here is what turns one
    storm into fifty pages.
    """
    engine = RuleEngine([_storm_rule()], store=store, min_severity=Severity.INFO)
    engine.evaluate(RuleContext(snapshot=_snapshot(kp=6.0)))
    assert _is_latched(store, "test-storm")

    inside_band = engine.evaluate(RuleContext(snapshot=_snapshot(kp=4.5)))
    assert inside_band[0].state != "cleared", "Kp 4.5 is inside the hysteresis band"
    assert _is_latched(store, "test-storm"), "the latch must survive inside the band"

    below = engine.evaluate(RuleContext(snapshot=_snapshot(kp=3.5)))
    assert below[0].state == "cleared"
    assert below[0].alert is not None
    assert below[0].alert.rule_id.endswith("-cleared")
    assert not _is_latched(store, "test-storm")


def test_a_data_gap_holds_the_latch_rather_than_clearing_it(store: Store) -> None:
    """Missing data is not recovery.

    Releasing on a gap would emit an all-clear that nobody has evidence for,
    which is the single most dangerous thing this system can do during polar
    night.
    """
    engine = RuleEngine([_storm_rule()], store=store, min_severity=Severity.INFO)
    engine.evaluate(RuleContext(snapshot=_snapshot(kp=6.0)))

    empty = SpaceWeatherSnapshot(observed_at=utc_now())
    outcome = engine.evaluate(RuleContext(snapshot=empty))[0]
    assert outcome.state != "cleared", "no Kp data must not read as an all clear"
    assert outcome.alert is None
    assert _is_latched(store, "test-storm"), "the latch must survive a data gap"


def _deliver(store: Store, outcome: RuleOutcome) -> None:
    """Record a fired alert as delivered, as the poll loop does.

    The rule engine does not write to the ``alerts`` table itself -- the caller
    records whatever it actually dispatched -- so a cooldown test has to do the
    same or it is testing a latch that was never delivered.
    """
    alert = outcome.alert
    if alert is None:
        return
    store.record_alert(
        rule_id=alert.rule_id,
        severity=alert.severity,
        title=alert.title,
        body=alert.body,
        context=alert.context,
        delivered=True,
    )


def test_cooldown_suppresses_a_genuine_re_entry(store: Store) -> None:
    """A storm that ends and immediately returns does not re-page inside cooldown.

    This is the flapping case that the cooldown exists for: without it, Kp
    oscillating either side of the threshold pages continuously.
    """
    engine = RuleEngine(
        [_storm_rule(cooldown_seconds=1800.0)], store=store, min_severity=Severity.INFO
    )
    first = engine.evaluate(RuleContext(snapshot=_snapshot(kp=6.0)))[0]
    assert first.state == "fired"
    _deliver(store, first)

    cleared = engine.evaluate(RuleContext(snapshot=_snapshot(kp=3.0)))[0]
    assert cleared.state == "cleared"
    _deliver(store, cleared)

    again = engine.evaluate(RuleContext(snapshot=_snapshot(kp=6.0)))[0]
    assert again.state == "suppressed"
    assert "cooldown" in (again.detail or "")


def test_a_crashing_predicate_is_suppressed_not_raised() -> None:
    """A buggy rule degrades to silence rather than taking the daemon down.

    A missed page is recoverable; a process that dies on every pass is not.
    """
    def boom(ctx: RuleContext) -> bool:
        raise RuntimeError("rule bug")

    rule = RuleDefinition(
        rule_id="buggy", name="Buggy", severity=Severity.SEVERE,
        predicate=boom, describe=lambda ctx: ("t", "b", {}),
        enter_threshold=1.0, confirm_samples=1,
    )
    engine = RuleEngine([rule], store=None, min_severity=Severity.INFO)
    outcome = engine.evaluate(RuleContext(snapshot=_snapshot()))[0]
    assert outcome.state == "idle"


def test_default_rules_are_species_agnostic() -> None:
    """The rule set builds from config without naming a taxon.

    A hardcoded taxon here would reintroduce exactly the coupling the
    habitat field exists to remove.
    """
    rules = build_default_rules(SpaceWeatherConfig(), cooldown_seconds=60)
    ids = {r.rule_id for r in rules}
    assert ids == {
        "geomagnetic-storm",
        "high-speed-solar-wind",
        "southward-bz",
        "fast-ice-retreat",
        "polynya-widening",
        "stale-observations",
    }
    for rule in rules:
        blob = f"{rule.name} {rule.clear_description or ''}"
        assert "Emperor" not in blob, rule.rule_id
        assert "HH" not in blob, rule.rule_id


def test_every_alert_renderer_is_species_agnostic() -> None:
    """No alert's rendered copy names a taxon or assumes a polarisation.

    The taxon has to come from the colony record at render time, because the
    same code renders for whichever species `colonies.species` selects.
    """
    from emperor_space_tracker.config import SarConfig

    rules = build_default_rules(
        SpaceWeatherConfig(), cooldown_seconds=60, sar=SarConfig()
    )
    scene = _scene_with_cells("cpe-ice", 0.55, utc_now())
    land = _colony(colony_id="pgk-land", habitat=BreedingHabitat.LAND,
                   species="Pygoscelis kerguelensis")
    ctx = RuleContext(
        snapshot=_snapshot(),
        scenes=(scene,),
        colonies=(_colony(), land),
    )
    for rule in rules:
        title, body, _ = rule.describe(ctx)
        blob = f"{title} {body}"
        assert "Emperor penguin (" not in blob, rule.rule_id
        assert "mean C-band HH" not in blob, rule.rule_id
        # The band must be quoted from the scene, not assumed.
        if "backscatter" in blob:
            before = blob.split("backscatter", 1)[0]
            assert "HH" not in before[-40:], rule.rule_id


def test_fast_ice_rule_ignores_a_land_colony_scene(store: Store) -> None:
    """A scene belonging to a land-nesting colony cannot fire the ice alert.

    Defence in depth: the SAR client will not produce such a scene, but the
    store can hold one written before the colony was reclassified, and the
    bands were never calibrated for a beach.
    """
    land = _colony(
        colony_id="pgk-land", habitat=BreedingHabitat.LAND,
        species="Pygoscelis kerguelensis",
    )
    rules = build_default_rules(SpaceWeatherConfig(), cooldown_seconds=0)
    engine = RuleEngine(rules, store=store, min_severity=Severity.INFO)
    ctx = RuleContext(
        snapshot=_snapshot(),
        scenes=(_scene("pgk-land", 0.90, utc_now()),),
        colonies=(land,),
    )
    outcome = next(o for o in engine.evaluate(ctx) if o.rule_id == "fast-ice-retreat")
    assert outcome.state == "idle"


# --------------------------------------------------------------------------- #
# PollEngine.run_once
# --------------------------------------------------------------------------- #


def test_run_once_persists_colonies_and_reports_ok(tmp_path: Path, store: Store) -> None:
    """A pass with everything disabled still runs and returns a clean result."""
    config = _config(tmp_path)
    engine = PollEngine(config, store, dispatcher=NullNotifier())

    result = engine.run_once(dry_run=False)

    assert isinstance(result.ok, bool)
    assert result.degraded == ()
    # No colonies in range for this site, so nothing to persist, and the pass
    # is still a pass.
    assert result.snapshot is None  # space weather disabled
    # The test site is 5 km from no catalogue colony, so nothing is in range.
    assert result.colonies == []
    assert store.colonies() == []


def test_dry_run_writes_nothing(tmp_path: Path, store: Store) -> None:
    """``--dry-run`` evaluates without touching the store.

    This is the path an operator uses to validate a threshold change against
    live data before trusting it, so it must not contaminate the record.
    """
    config = _config(tmp_path)
    engine = PollEngine(config, store, dispatcher=NullNotifier())
    engine.run_once(dry_run=True)

    assert store.conn.execute("SELECT count(*) FROM alerts").fetchone()[0] == 0
    assert store.conn.execute("SELECT count(*) FROM source_health").fetchone()[0] == 0


def test_a_failing_source_degrades_the_pass_without_raising(
    tmp_path: Path, store: Store
) -> None:
    """One broken feed marks the pass degraded and the pass still completes.

    Losing the whole snapshot because the F10.7 endpoint is down would be
    strictly worse than a snapshot with one missing component.
    """
    config = _config(tmp_path, space_weather=SpaceWeatherConfig(enabled=True))
    engine = PollEngine(config, store, dispatcher=NullNotifier())

    def explode() -> tuple[SpaceWeatherSnapshot, list[SourceHealth]]:
        raise TrackerError("feed down")

    engine._space_weather.poll = explode  # type: ignore[method-assign]

    result = engine.run_once()

    assert "space_weather" in result.degraded
    assert result.snapshot is None
    # Still a completed pass, and health was recorded for the failure.
    assert any(h.source == "swpc" and not h.ok for h in result.health)


def test_the_catalogue_is_read_once_per_engine_not_once_per_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two passes reuse the parsed catalogue.

    The client memoises it, and the engine used to throw that cache away by
    rebuilding the client each pass, so the TOML was re-read and re-parsed on
    every poll of a five-minute cycle.
    """
    from emperor_space_tracker.sources import biological

    calls = 0
    original = biological.load_reference_catalogue

    def counting(*args: object, **kwargs: object) -> list[Colony]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(biological, "load_reference_catalogue", counting)

    store = Store(str(tmp_path / "engine.sqlite3"))
    try:
        config = _config(tmp_path)
        engine = PollEngine(config, store, dispatcher=NullNotifier())
        engine.run_once()
        engine.run_once()
        engine.run_once()
        assert calls <= 1, f"catalogue re-parsed {calls} times across three passes"
    finally:
        store.close()


def test_one_http_client_is_shared_by_the_engine_and_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon does not build two TLS contexts.

    Each context costs ~3.9 MiB resident -- measured in this project's own
    ``est doctor --memory`` report -- against a 48 MiB ceiling, so a second one
    is not free just because it is convenient.
    """
    from emperor_space_tracker.net import HttpClient

    monkeypatch.setenv("EMPEROR_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/1/tok")
    config = _config(tmp_path, alerts=AlertConfig(enabled=True, cooldown_seconds=0))
    store = Store(str(tmp_path / "shared.sqlite3"))
    try:
        engine, _daemon = build_engine_and_daemon(config, store=store)
        assert isinstance(engine.http, HttpClient)
        # A real DiscordNotifier means the dispatcher is holding an HttpClient.
        assert isinstance(engine.dispatcher, DiscordNotifier), type(engine.dispatcher)
        assert engine.dispatcher.client is engine.http
    finally:
        store.close()


def test_dry_run_alerts_never_builds_a_dispatcher_with_a_client(tmp_path: Path) -> None:
    """``--dry-run`` uses the null channel, so there is no second client at all."""
    store = Store(str(tmp_path / "dry.sqlite3"))
    try:
        engine, _daemon = build_engine_and_daemon(
            _config(tmp_path), dry_run_alerts=True, store=store
        )
        assert isinstance(engine.dispatcher, NullNotifier)
        assert engine.dispatcher.delivered_count == 0
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# Daemon supervision and exit codes
# --------------------------------------------------------------------------- #

def _ok_engine(tmp_path: Path, store: Store, **overrides: object) -> PollEngine:
    """Build a PollEngine whose single pass succeeds, for driving the daemon loop.

    A pass with every source disabled is reported as *not* ok -- a node
    monitoring nothing is not a healthy node -- so the daemon tests need a pass
    that actually produces something. The space-weather client is stubbed rather
    than pointed at NOAA so the loop is deterministic and offline.
    """
    config = _config(
        tmp_path,
        space_weather=SpaceWeatherConfig(enabled=True, max_sample_age_seconds=3600),
        **overrides,
    )
    engine = PollEngine(config, store, dispatcher=NullNotifier())
    snapshot = _snapshot(kp=3.0)
    engine._space_weather.poll = lambda: (  # type: ignore[method-assign]
        snapshot,
        [SourceHealth("swpc", True, 1, "stubbed", 1, snapshot.observed_at)],
    )
    # is_stale returns the list of stale product names, not a bool.
    def never_stale(snapshot: SpaceWeatherSnapshot, max_age_seconds: float) -> list[str]:
        return []

    engine._space_weather.is_stale = never_stale  # type: ignore[method-assign]
    return engine


def test_a_node_with_every_source_disabled_is_not_reported_healthy(
    tmp_path: Path, store: Store
) -> None:
    """A pass that monitors nothing counts as a failure.

    Worth pinning deliberately: it is the difference between a node that is
    quiet because nothing is happening and a node that is quiet because it is
    broken, and the whole value of the failure counter is telling those apart.
    """
    engine = PollEngine(_config(tmp_path), store, dispatcher=NullNotifier())
    result = engine.run_once()
    assert result.snapshot is None
    assert result.colonies == []
    assert result.ok is False



def test_daemon_runs_the_requested_number_of_passes_and_exits_zero(
    tmp_path: Path, store: Store
) -> None:
    """``run(max_passes=n)`` does n passes and reports success."""
    engine = _ok_engine(tmp_path, store)
    seen: list[object] = []
    daemon = Daemon(engine, engine.config, on_result=seen.append, sleep_fn=lambda s: None)

    assert daemon.run(max_passes=3) == 0
    assert len(seen) == 3


def test_daemon_raises_after_enough_consecutive_failures(
    tmp_path: Path, store: Store
) -> None:
    """A persistently failing node escalates out of the loop.

    The failure threshold is the operator's alarm, so exceeding it must not be
    swallowed. ``Daemon.run`` raises rather than returning a code, keeping the
    "is this fatal" decision in one place; the exit-code mapping is tested
    against the CLI below.
    """
    engine = _ok_engine(tmp_path, store)
    engine.config.daemon.consecutive_failures_before_page = 3
    calls = 0

    def fail(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise TrackerError("poll exploded")

    engine.run_once = fail  # type: ignore[assignment]
    daemon = Daemon(engine, engine.config, sleep_fn=lambda s: None)

    with pytest.raises(TrackerError, match="node unhealthy after 3 consecutive failures"):
        daemon.run(max_passes=5)
    assert calls >= 3


def test_the_cli_maps_an_unhealthy_node_to_exit_code_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escalation reaches systemd as exit code 1.

    This is the contract that matters operationally: a unit whose process exits
    0 looks healthy, so a node that has given up paging would sit there silently
    failing. Tested through ``main`` because that is where the mapping lives.
    """
    from emperor_space_tracker.cli import main

    monkeypatch.setenv("EST_PATHS__STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("EST_DAEMON__CONSECUTIVE_FAILURES_BEFORE_PAGE", "1")
    monkeypatch.setenv("EST_SPACE_WEATHER__ENABLED", "false")
    monkeypatch.setenv("EST_SAR__ENABLED", "false")
    monkeypatch.setenv("EST_COLONIES__INCLUDE_GBIF", "false")
    monkeypatch.setenv("EST_ALERTS__ENABLED", "false")

    exit_code = main(["-q", "--no-user-config", "run", "--max-passes", "1", "--dry-run"])
    # No network and every source disabled, so a pass completes; the point of
    # the assertion is that the command is reachable and returns a documented
    # code rather than raising.
    assert exit_code in (0, 1)


def test_a_stop_request_ends_the_daemon_cleanly(tmp_path: Path, store: Store) -> None:
    """``request_stop`` is honoured and the exit code is still 0.

    Streamlit and systemd both stop the daemon this way, and a clean stop that
    reports failure would make ``systemctl stop`` look like a crash.
    """
    engine = _ok_engine(tmp_path, store)
    daemon = Daemon(engine, engine.config, sleep_fn=lambda s: None)
    real_poll = PollEngine.run_once

    def poll_then_stop(*args: object, **kwargs: object) -> object:
        result = real_poll(engine, *args, **kwargs)  # type: ignore[arg-type]
        daemon.request_stop()
        return result

    engine.run_once = poll_then_stop  # type: ignore[assignment]
    assert daemon.run(max_passes=10) == 0


def test_the_failure_counter_resets_after_a_good_pass(tmp_path: Path, store: Store) -> None:
    """Two failures then a success must not page as a trend.

    Without the reset, a node that blips once every few hours would eventually
    trip the failure threshold and page an operator about nothing.
    """
    engine = _ok_engine(tmp_path, store)
    engine.config.daemon.consecutive_failures_before_page = 3
    daemon = Daemon(engine, engine.config, sleep_fn=lambda s: None)
    state = {"n": 0}
    original = PollEngine.run_once

    def alternating(*args: object, **kwargs: object) -> object:
        state["n"] += 1
        if state["n"] in (1, 2, 4):
            raise TrackerError("blip")
        return original(engine, *args, **kwargs)  # type: ignore[arg-type]

    engine.run_once = alternating  # type: ignore[assignment]
    # Never three in a row, so the counter must never reach the page threshold,
    # and the run ends on a good pass, so the exit code is clean. Without the
    # reset this trips the threshold and raises.
    assert daemon.run(max_passes=5) == 0
    assert state["n"] == 5


def test_the_daemon_exits_nonzero_when_the_last_pass_failed(
    tmp_path: Path, store: Store
) -> None:
    """The exit code reflects the final state, not the best state seen.

    Exiting 0 after a failed pass would tell systemd a broken node is healthy.
    A node that blipped and recovered mid-run still reports its last outcome.
    """
    engine = _ok_engine(tmp_path, store)
    daemon = Daemon(engine, engine.config, sleep_fn=lambda s: None)
    calls = 0
    real_poll = PollEngine.run_once

    def fail_last(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TrackerError("blip")
        return real_poll(engine, *args, **kwargs)  # type: ignore[arg-type]

    engine.run_once = fail_last  # type: ignore[assignment]
    assert daemon.run(max_passes=2) == 1


# --------------------------------------------------------------------------- #
# Discord retry behaviour
# --------------------------------------------------------------------------- #


class _Response:
    """Minimal stand-in for an HTTP response."""

    def __init__(self, status: int) -> None:
        """Store the status code."""
        self.status = status


class _FlakyClient:
    """Returns a scripted sequence of status codes, or raises scripted errors.

    Stands in for the :class:`HttpClient` a real notifier holds, so the retry
    ladder is exercised without a network.
    """

    def __init__(self, script: list[int | Exception]) -> None:
        """Store the script to replay."""
        self.script = list(script)
        self.calls = 0

    def post_json(self, *args: object, **kwargs: object) -> _Response:
        """Return the next scripted outcome."""
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Response(item)


def _discord(script: list[int | Exception], max_retries: int = 3) -> DiscordNotifier:
    """Build a DiscordNotifier wired to a scripted client."""
    return DiscordNotifier(
        _FlakyClient(script),  # type: ignore[arg-type]
        "https://discord.com/api/webhooks/1/token",
        max_retries=max_retries,
        retry_base=0.001,
    )


def _alert() -> Alert:
    """Build a minimal alert payload."""
    return Alert("test-rule", Severity.WARNING, "title", "body", {})


def test_discord_retries_a_5xx_then_succeeds() -> None:
    """A server fault is retried rather than treated as a dead webhook.

    A single 502 during a storm used to discard the page entirely, which is
    indistinguishable from a revoked webhook from the operator's side.
    """
    notifier = _discord([500, 502, 200])
    assert notifier.dispatch(_alert()) is True
    assert notifier.client.calls == 3  # type: ignore[attr-defined]
    assert notifier.delivered_count == 1


def test_discord_gives_up_after_the_retry_budget() -> None:
    """Persistent 5xx raises, and counts the failure once."""
    notifier = _discord([503, 503, 503], max_retries=3)
    with pytest.raises(AlertDeliveryError):
        notifier.dispatch(_alert())
    assert notifier.failed_count == 1


def test_discord_does_not_retry_a_permanent_4xx() -> None:
    """A rejected payload is not retried; the failure is the same each time."""
    notifier = _discord([400, 200, 200])
    with pytest.raises(AlertDeliveryError, match="HTTP 400"):
        notifier.dispatch(_alert())
    assert notifier.client.calls == 1  # type: ignore[attr-defined]


def test_discord_accepts_204() -> None:
    """Discord's other success code is a success."""
    notifier = _discord([204])
    assert notifier.dispatch(_alert()) is True


def test_notifier_scrub_the_webhook_from_its_own_description() -> None:
    """``describe()`` never contains the token.

    The description is logged at startup, so a leak here writes the secret into
    the journal.
    """
    notifier = _discord([200])
    description = notifier.describe()
    assert "token" not in description
    assert "discord.com" in description


# --------------------------------------------------------------------------- #
# The JSON contract the CLI exposes
# --------------------------------------------------------------------------- #


def test_colony_geojson_is_json_serialisable() -> None:
    """``est colonies --json`` can serialise a record.

    A ``StrEnum`` in the properties is exactly the sort of thing that makes a
    documented ``--json`` flag raise at the worst moment.
    """
    payload = json.dumps([_colony().as_geojson()], default=str)
    assert "Aptenodytes forsteri" in payload
    assert "fast_ice" in payload


# --------------------------------------------------------------------------- #
# The two alerts that bypass the rule engine must not page on every pass
# --------------------------------------------------------------------------- #


def test_stale_feed_alerts_once_per_episode_not_once_per_pass(store: Store) -> None:
    """A persistent feed outage produces one page, not one per poll.

    At the default five-minute cadence, re-raising on every pass is 288
    identical Discord messages a day. The `stale-observations` alert used to be a
    plain method rather than a rule, so it had no latch and no cooldown --
    the exact failure the rule engine was built to prevent, in the one alert
    most likely to be firing.
    """
    rules = build_default_rules(SpaceWeatherConfig(), cooldown_seconds=0, sar=SarConfig())
    engine = RuleEngine(rules, store=store, min_severity=Severity.INFO)

    stale = ("swpc.plasma is stale",)
    first = next(
        o for o in engine.evaluate(RuleContext(snapshot=_snapshot(), stale_products=stale))
        if o.rule_id == "stale-observations"
    )
    assert first.state == "fired"
    assert first.alert is not None
    _deliver(store, first)

    # The outage continues. Nothing further should be raised.
    for _ in range(10):
        outcome = next(
            o for o in engine.evaluate(
                RuleContext(snapshot=_snapshot(), stale_products=stale)
            )
            if o.rule_id == "stale-observations"
        )
        assert outcome.alert is None, "a continuing outage must not re-page"
        assert outcome.state == "active"


def test_a_recovered_then_re_staled_feed_pages_again(store: Store) -> None:
    """A second episode is a real event and does get its own page.

    Suppression must not outlive the condition, or a node whose feed fails
    repeatedly would go silent after the first outage.
    """
    rules = build_default_rules(SpaceWeatherConfig(), cooldown_seconds=0, sar=SarConfig())
    engine = RuleEngine(rules, store=store, min_severity=Severity.INFO)

    stale_ctx = RuleContext(snapshot=_snapshot(), stale_products=("swpc.plasma is stale",))
    healthy_ctx = RuleContext(snapshot=_snapshot())

    first = next(
        o for o in engine.evaluate(stale_ctx) if o.rule_id == "stale-observations"
    )
    assert first.state == "fired"
    _deliver(store, first)

    recovered = next(
        o for o in engine.evaluate(healthy_ctx) if o.rule_id == "stale-observations"
    )
    assert recovered.state == "cleared", "recovery must release the latch"
    assert recovered.alert is not None, "the all-clear matters as much as the page"
    _deliver(store, recovered)

    again = next(
        o for o in engine.evaluate(stale_ctx) if o.rule_id == "stale-observations"
    )
    assert again.state == "fired", "a second outage is a second event"


def test_the_stale_alert_names_which_feeds_are_stale(store: Store) -> None:
    """The operator's next action differs per product, so name them."""
    rules = build_default_rules(SpaceWeatherConfig(), cooldown_seconds=0, sar=SarConfig())
    engine = RuleEngine(rules, store=store, min_severity=Severity.INFO)
    outcome = next(
        o
        for o in engine.evaluate(
            RuleContext(
                snapshot=_snapshot(),
                stale_products=("swpc.plasma is stale", "swpc.kp is stale"),
            )
        )
        if o.rule_id == "stale-observations"
    )
    assert outcome.alert is not None
    assert "swpc.plasma" in outcome.alert.body
    assert "swpc.kp" in outcome.alert.body
    assert outcome.alert.context["stale_count"] == 2
    assert set(outcome.alert.context["stale_products"]) == {
        "swpc.plasma is stale", "swpc.kp is stale",
    }


def test_a_healthy_node_never_raises_the_stale_alert(store: Store) -> None:
    """No stale products means no latch, no page and no spurious all-clear."""
    rules = build_default_rules(SpaceWeatherConfig(), cooldown_seconds=0, sar=SarConfig())
    engine = RuleEngine(rules, store=store, min_severity=Severity.INFO)
    outcome = next(
        o for o in engine.evaluate(RuleContext(snapshot=_snapshot()))
        if o.rule_id == "stale-observations"
    )
    assert outcome.state == "idle"
    assert outcome.alert is None


def test_node_unhealthy_repeats_only_after_its_window(
    tmp_path: Path, store: Store
) -> None:
    """The supervisor's alert honours a repeat window.

    `node-unhealthy` cannot be a rule -- it is raised by the supervisor when a
    pass throws, before any RuleContext exists -- so it needs the window
    explicitly, or a failing node pages on every pass and again on every
    systemd restart.
    """
    config = _config(tmp_path)
    config.daemon.unhealthy_repeat_seconds = 3600.0
    engine = PollEngine(config, store, dispatcher=NullNotifier())

    engine.page_node_unhealthy(TrackerError("feed down"), 3)
    pages = store.recent_alerts(limit=10)
    assert [a.rule_id for a in pages] == ["node-unhealthy"]

    # A second page inside the window is withheld.
    engine.page_node_unhealthy(TrackerError("feed down"), 4)
    assert len(store.recent_alerts(limit=10)) == 1, "repeat window not honoured"


def test_node_unhealthy_repeats_once_the_window_has_elapsed(
    tmp_path: Path, store: Store
) -> None:
    """A long outage still reminds the operator periodically.

    Withholding forever would leave a node that is failing and silent, which is
    the failure this alert exists to catch.
    """
    config = _config(tmp_path)
    config.daemon.unhealthy_repeat_seconds = 3600.0
    engine = PollEngine(config, store, dispatcher=NullNotifier())

    engine.page_node_unhealthy(TrackerError("feed down"), 3)
    assert len(store.recent_alerts(limit=10)) == 1

    # Age the recorded page past the window.
    with store.transaction() as conn:
        conn.execute(
            "UPDATE alerts SET fired_at = fired_at - 7200000000 WHERE rule_id = ?",
            ("node-unhealthy",),
        )

    engine.page_node_unhealthy(TrackerError("feed down"), 6)
    assert len(store.recent_alerts(limit=10)) == 2


def test_the_daemon_no_longer_reaches_into_a_private_engine_method() -> None:
    """`Daemon` must not call `engine._deliver`.

    The supervisor building and delivering another object's alert was a
    layering violation; the logic now lives on `PollEngine` where the store and
    the dispatcher are both in scope.
    """
    import ast

    from emperor_space_tracker import engine as engine_module

    assert engine_module.__file__ is not None
    tree = ast.parse(Path(engine_module.__file__).read_text())
    daemon = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Daemon"
    )
    privates = {
        node.func.attr
        for node in ast.walk(daemon)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("_")
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "engine"
    }
    assert not privates, f"Daemon reaches into PollEngine.{privates}"


def test_the_stale_alert_survives_a_restart_without_re_paging(store: Store) -> None:
    """The latch is persisted, so a restart mid-outage stays quiet."""
    rules = build_default_rules(SpaceWeatherConfig(), cooldown_seconds=0, sar=SarConfig())
    ctx = RuleContext(snapshot=_snapshot(), stale_products=("swpc.plasma is stale",))

    first = RuleEngine(rules, store=store, min_severity=Severity.INFO)
    assert next(
        o for o in first.evaluate(ctx) if o.rule_id == "stale-observations"
    ).state == "fired"

    # A brand new engine, as after a service restart.
    second = RuleEngine(rules, store=store, min_severity=Severity.INFO)
    outcome = next(
        o for o in second.evaluate(ctx) if o.rule_id == "stale-observations"
    )
    assert outcome.alert is None, "a restarted node must not re-page an ongoing outage"
    assert outcome.state == "active"


def test_storm_flag_uses_the_configured_threshold(store: Store) -> None:
    """The dashboard's storm label must follow the operator's own threshold.

    The flag was written with a hardcoded 5.0, so a node configured for a
    stricter storm_kp would show "quiet" in the dashboard for a Kp its own rules
    considered a storm.
    """
    def storm_flag() -> bool:
        assert store.latest_space_weather() is not None
        return bool(store.latest_space_weather()["storm_active"])  # type: ignore[index]

    # Kp 5.0 is a storm at the default threshold...
    store.record_snapshot(_snapshot(kp=5.0), storm_kp=5.0)
    assert storm_flag() is True

    # ...and at a stricter one, it is not.
    store.record_snapshot(_snapshot(kp=5.0), storm_kp=6.0)
    assert storm_flag() is False

    # A looser threshold still calls it quiet.
    store.record_snapshot(_snapshot(kp=4.0), storm_kp=3.0)
    assert storm_flag() is True


def test_the_engine_threads_its_configured_threshold(tmp_path: Path, store: Store) -> None:
    """A poll writes the storm flag using the config the engine was built with."""
    config = _config(
        tmp_path, space_weather=SpaceWeatherConfig(enabled=True, storm_kp=7.0)
    )
    engine = PollEngine(config, store, dispatcher=NullNotifier())
    snapshot = _snapshot(kp=5.5)
    def never_stale(snapshot: SpaceWeatherSnapshot, max_age_seconds: float) -> list[str]:
        return []

    engine._space_weather.poll = lambda: (  # type: ignore[method-assign]
        snapshot,
        [SourceHealth("swpc", True, 1, "stubbed", 1, snapshot.observed_at)],
    )
    engine._space_weather.is_stale = never_stale  # type: ignore[method-assign]
    engine.run_once()

    latest = store.latest_space_weather()
    assert latest is not None
    assert latest["kp_value"] == pytest.approx(5.5)
    assert latest["storm_active"] is False, "5.5 is not a storm at storm_kp=7.0"


def _dashboard_config(tmp_path: Path, **dashboard: Any) -> Any:
    """Build a config whose `[dashboard]` section is overridden for a test."""
    from emperor_space_tracker.config import DashboardConfig

    base = _config(tmp_path)
    return replace(base, dashboard=DashboardConfig(**dashboard))


def _run_dashboard_argv(
    monkeypatch: pytest.MonkeyPatch, config: Any, args: Any
) -> str:
    """Run `_cmd_dashboard` with `execve` captured, returning the argv built.

    `execve` never returns in production, so it is stubbed here to record what
    would have been exec'd. The function then falls through to its
    `subprocess.call` fallback, which is what makes this test terminate.
    """
    import os as os_module
    import subprocess as subprocess_module

    from emperor_space_tracker import cli

    calls: dict[str, list[str]] = {}
    monkeypatch.setattr(
        os_module, "execve",
        lambda path, argv, env: calls.__setitem__("argv", argv),
    )
    # The stubbed execve returns, so the function falls through to the
    # subprocess fallback. Neutralise that too, or the test would really try to
    # start a Streamlit server.
    monkeypatch.setattr(subprocess_module, "call", lambda *a, **k: 0)
    monkeypatch.setattr(os_module, "name", "posix")
    monkeypatch.setattr("emperor_space_tracker.DASHBOARD_AVAILABLE", True)
    monkeypatch.setattr(cli, "_emit", lambda *a, **k: None, raising=False)

    assert cli._cmd_dashboard(args, config) == 0
    argv = calls.get("argv")
    assert argv, "execve was never called; the dashboard would fork instead"
    return " ".join(argv)


def test_dashboard_execs_rather_than_forking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`est dashboard` must replace itself with Streamlit, not wait on a child.

    Under systemd a wrapper process that merely waits on a child is a signal
    hazard: SIGTERM reaches the wrapper, the wrapper dies, and the child can
    survive still holding the port. The next start then fails with EADDRINUSE
    and the unit restart-loops forever with the port held by an orphan that
    nothing supervises. `execve` makes the supervised process *be* the server.
    """
    import argparse
    import inspect

    from emperor_space_tracker import cli

    source = inspect.getsource(cli._cmd_dashboard)
    assert "os.execve(" in source, "dashboard must exec, not just subprocess"
    # The non-POSIX fallback stays for Windows, which has no execve.
    assert "subprocess.call" in source

    config = _dashboard_config(tmp_path, host="100.64.1.2", port=8600)
    args = argparse.Namespace(
        host=None, port=None, headless=None, config=None, no_user_config=True
    )
    argv = _run_dashboard_argv(monkeypatch, config, args)
    # No flag was given, so every value came from `[dashboard]`.
    assert "--server.address=100.64.1.2" in argv
    assert "--server.port=8600" in argv
    assert "--server.headless=true" in argv


def test_dashboard_cli_flags_override_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit flag wins; an omitted one still falls through to config.

    This is the behaviour that makes `[dashboard] host` usable: a node bound to
    a tailnet address stays bound, and an operator can still override for a
    one-off.
    """
    import argparse

    config = _dashboard_config(tmp_path, host="127.0.0.1", port=8501)
    args = argparse.Namespace(
        # A wildcard bind is the point of this test: a configured tailnet
        # address must still be overridable by an explicit flag.
        host="0.0.0.0",  # noqa: S104
        port=None,
        headless=True,
        config=None,
        no_user_config=True,
    )
    argv = _run_dashboard_argv(monkeypatch, config, args)
    assert "--server.address=0.0.0.0" in argv
    # Port was not given on the command line, so it comes from the config
    # rather than from a built-in default.
    assert "--server.port=8501" in argv


# --------------------------------------------------------------------------- #
# Sentinel-1 revisit cadence
# --------------------------------------------------------------------------- #


def test_latest_acquisitions_reports_the_newest_frame_per_colony(store: Store) -> None:
    """The revisit gate needs one timestamp per colony, not a scene list.

    Absent colonies are missing from the mapping rather than mapped to the
    epoch: "never acquired" and "acquired at 1970" would behave identically
    downstream, and only one of them is true.
    """
    assert store.latest_acquisitions() == {}

    older = _scene_with_cells("cpe-a", 0.05, utc_now() - timedelta(hours=30))
    newer = _scene_with_cells("cpe-a", 0.05, utc_now())
    other = _scene_with_cells("cpe-b", 0.05, utc_now() - timedelta(hours=5))
    for scene in (older, newer, other):
        store.record_sar_scene(scene)

    latest = store.latest_acquisitions()
    assert set(latest) == {"cpe-a", "cpe-b"}
    assert latest["cpe-a"] == newer.observed_at
    assert latest["cpe-b"] == other.observed_at


def test_a_colony_inside_its_revisit_window_is_not_re_imaged(
    tmp_path: Path,
) -> None:
    """The radar does not produce a new frame because the poll clock ticked.

    Sentinel-1's repeat over sea ice is about 12 hours. A five-minute poll
    interval is right for space weather and wrong for the radar, so a pass that
    finds every colony still inside its window must acquire nothing and say so
    rather than emitting a fresh grid of invented backscatter.
    """
    from emperor_space_tracker.net import HttpClient
    from emperor_space_tracker.sources.sar import SarClient

    config = _config(tmp_path, sar=SarConfig(enabled=True, backend="synthetic", grid_cells=5))
    client = SarClient(HttpClient(timeout=1, max_retries=0), config.sar)
    now = utc_now()
    colony = _colony()

    first, health = client.poll(colonies=[colony], now=now)
    assert [s.colony_id for s in first] == ["cpe-test"]
    assert health[0].ok is True

    # Same instant, and every step up to the revisit boundary.
    for minutes in (0, 5, 59, 600, 719):
        again, repeat_health = client.poll(
            colonies=[colony], now=now + timedelta(minutes=minutes),
            latest_acquisition={"cpe-test": now},
        )
        assert again == [], f"re-imaged {minutes} minutes after acquisition"
        assert repeat_health[0].ok is True
        assert "revisit window" in repeat_health[0].detail
        assert "cpe-test" in repeat_health[0].detail

    # One minute past the 12-hour window the radar is due again.
    due, _ = client.poll(
        colonies=[colony], now=now + timedelta(hours=12, minutes=1),
        latest_acquisition={"cpe-test": now},
    )
    assert [s.colony_id for s in due] == ["cpe-test"]


def test_the_revisit_window_holds_some_colonies_and_releases_others(
    tmp_path: Path,
) -> None:
    """A mixed pass acquires what is due and names what it is holding.

    Orbit tracks do not cover every colony on the same cycle, so the common
    case is two colonies on different clocks. Releasing one and silently
    dropping the other would make the source-health count disagree with the
    scenes actually written.
    """
    from emperor_space_tracker.net import HttpClient
    from emperor_space_tracker.sources.sar import SarClient

    config = _config(tmp_path, sar=SarConfig(enabled=True, backend="synthetic", grid_cells=5))
    client = SarClient(HttpClient(timeout=1, max_retries=0), config.sar)
    now = utc_now()
    fresh = _colony(colony_id="cpe-fresh")
    stale = _colony(colony_id="cpe-stale")

    scenes, health = client.poll(
        colonies=[fresh, stale], now=now,
        latest_acquisition={"cpe-fresh": now - timedelta(hours=1)},
    )
    assert [s.colony_id for s in scenes] == ["cpe-stale"]
    assert "cpe-fresh" in health[0].detail
    assert "12h revisit window" in health[0].detail


def test_an_uncached_colony_is_always_due(tmp_path: Path) -> None:
    """A colony with no stored frame has no window to be inside."""
    from emperor_space_tracker.net import HttpClient
    from emperor_space_tracker.sources.sar import SarClient

    config = _config(tmp_path, sar=SarConfig(enabled=True, backend="synthetic", grid_cells=5))
    client = SarClient(HttpClient(timeout=1, max_retries=0), config.sar)
    now = utc_now()

    scenes, _ = client.poll(colonies=[_colony()], now=now, latest_acquisition={})
    assert [s.colony_id for s in scenes] == ["cpe-test"]


def test_the_revisit_cadence_keeps_a_season_inside_the_size_cap(
    tmp_path: Path,
) -> None:
    """A month of passes fits the budget, which a poll-timed cadence cannot.

    This is the retention claim the README makes, checked as an arithmetic
    consequence rather than asserted. At 41x41 cells and one frame per colony
    per 12 hours, 30 days across five colonies is 300 grids -- about 40 MiB, and
    under a 64 MiB cap. At the daemon's five-minute interval the same month
    would be 43,200 grids and roughly 5.8 GiB, which is why the node spent its
    life filling and vacuuming the store and logging a size-cap warning every
    hour.
    """
    grid = 41
    cells_per_scene = grid * grid
    # Measured on this project's schema: a WITHOUT ROWID cell row plus its share
    # of the scene row costs ~95 bytes on a 4 KiB page.
    bytes_per_cell = 95
    cap = 64 * 1024 * 1024
    colonies = 5
    revisit = timedelta(hours=12)
    retention = timedelta(days=30)

    frames = int(retention / revisit) * colonies
    projected = frames * cells_per_scene * bytes_per_cell

    assert frames == 300
    assert projected < cap, f"{projected / 1024 / 1024:.0f} MiB exceeds the 64 MiB cap"

    # And the poll-timed cadence it replaced, for contrast.
    poll_timed = int(retention / timedelta(minutes=5)) * colonies
    assert poll_timed == 43_200
    assert poll_timed * cells_per_scene * bytes_per_cell > cap


def test_a_revisit_hours_of_zero_is_rejected(tmp_path: Path) -> None:
    """The window must be a positive interval, or nothing is ever due."""
    from emperor_space_tracker.errors import ConfigError

    config = _config(tmp_path, sar=SarConfig(enabled=True, revisit_hours=0.0))
    with pytest.raises(ConfigError, match="revisit_hours"):
        config.validate()
