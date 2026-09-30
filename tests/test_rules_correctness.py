"""Tests pinning the correctness-fix batch: alert copy, hysteresis, cooldowns.

Every behaviour here was previously a silent wrong answer: a mislabelled
storm scale, a hysteresis band that never engaged, a config value that never
reached the rules, a layout that reshuffled every rerun, a footprint sized
from the wrong knob, and a decompression fallback that leaked raw exceptions.
"""

from __future__ import annotations

import logging
import sys
import types
from datetime import timedelta
from types import SimpleNamespace

import pytest

from emperor_space_tracker.alerts.rules import (
    RuleContext,
    RuleDefinition,
    RuleEngine,
    RuleOutcome,
    _g_scale,
    build_default_rules,
)
from emperor_space_tracker.config import SarConfig, SpaceWeatherConfig
from emperor_space_tracker.errors import SourceUnavailableError
from emperor_space_tracker.models import (
    BreedingHabitat,
    Colony,
    InterplanetaryMagneticField,
    KpIndex,
    SarScene,
    Severity,
    SolarWindPlasma,
    SpaceWeatherSnapshot,
    utc_now,
)
from emperor_space_tracker.net import HttpClient
from emperor_space_tracker.sources.sar import SarClient

# --------------------------------------------------------------------------- #
# G_SCALE: NOAA labels (G1 = Kp 5 … G5 = Kp 9)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kp", "expected"),
    [
        (4.9, "below storm threshold"),
        (5.0, "G1 minor storm"),
        (5.5, "G1 minor storm"),
        (6.0, "G2 moderate storm"),
        (7.0, "G3 strong storm"),
        (8.0, "G4 severe storm"),
        (9.0, "G5 extreme storm"),
    ],
)
def test_g_scale_matches_noaa(kp: float, expected: str) -> None:
    """Each Kp boundary maps to its NOAA G-scale, not a rotated label."""
    assert _g_scale(kp) == expected


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _snapshot(
    *,
    kp: float | None = None,
    speed: float | None = None,
    bz: float | None = None,
) -> SpaceWeatherSnapshot:
    """Build a snapshot with only the requested products present."""
    kp_rec = (
        KpIndex(observed_at=utc_now(), kp_index=None, estimated_kp=kp)
        if kp is not None
        else None
    )
    plasma = (
        SolarWindPlasma(
            observed_at=utc_now(),
            source="TEST",
            speed_kms=speed,
            density_per_cm3=5.0,
            temperature_k=100000.0,
            active=True,
        )
        if speed is not None
        else None
    )
    field = (
        InterplanetaryMagneticField(
            observed_at=utc_now(),
            source="TEST",
            bt_nt=5.0,
            bz_gsm_nt=bz,
            by_gsm_nt=0.0,
            density_pct=None,
            active=True,
        )
        if bz is not None
        else None
    )
    return SpaceWeatherSnapshot(
        observed_at=utc_now(), plasma=plasma, magnetic_field=field, kp=kp_rec
    )


def _scene(water: float, colony_id: str = "test-colony") -> SarScene:
    """Build a minimal scene with the given open-water fraction."""
    return SarScene(
        scene_id=f"test-{water}",
        observed_at=utc_now(),
        platform="SENTINEL-1",
        orbit_type="ASCENDING",
        polarisation="VV",
        incidence_angle_deg=None,
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


def _rules() -> list[RuleDefinition]:
    """Default rules with single-sample confirmation for deterministic tests."""
    config = SpaceWeatherConfig(consecutive_samples=1)
    return build_default_rules(config)


def _latched(rule_id: str, context: RuleContext) -> RuleEngine:
    """Return an engine with ``rule_id`` latched via one firing pass."""
    engine = RuleEngine(_rules(), store=None)
    outcomes = engine.evaluate(context)
    fired = next(o for o in outcomes if o.rule_id == rule_id)
    assert fired.state == "fired", f"setup pass did not fire {rule_id}: {fired}"
    return engine


def _outcome(engine: RuleEngine, rule_id: str, context: RuleContext) -> RuleOutcome:
    """Evaluate one pass and return the outcome for ``rule_id``."""
    (found,) = [o for o in engine.evaluate(context) if o.rule_id == rule_id]
    return found


# --------------------------------------------------------------------------- #
# Cooldown wiring: [alerts] cooldown_seconds reaches the space-weather rules
# --------------------------------------------------------------------------- #


def test_cooldown_seconds_reaches_space_weather_rules() -> None:
    """A configured cooldown propagates; the ice rule keeps its own."""
    rules = build_default_rules(SpaceWeatherConfig(), cooldown_seconds=999.0)
    space_rules = [r for r in rules if r.rule_id != "fast-ice-retreat"]
    assert space_rules, "expected space-weather rules"
    assert {r.cooldown_seconds for r in space_rules} == {999.0}
    ice = next(r for r in rules if r.rule_id == "fast-ice-retreat")
    assert ice.cooldown_seconds == 3600.0


def test_engine_threads_alerts_cooldown() -> None:
    """PollEngine builds its rules from the effective configuration."""
    import tempfile
    from pathlib import Path

    from emperor_space_tracker.config import load_config
    from emperor_space_tracker.engine import PollEngine
    from emperor_space_tracker.store import Store

    config = load_config(use_user_config=False)
    config.alerts.cooldown_seconds = 1234.0
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "test.sqlite3")
        try:
            engine = PollEngine(config, store)
            by_id = {r.rule_id: r for r in engine.rule_engine.rules}
            assert by_id["geomagnetic-storm"].cooldown_seconds == 1234.0
            assert by_id["fast-ice-retreat"].cooldown_seconds == 3600.0
        finally:
            store.close()


# --------------------------------------------------------------------------- #
# Hysteresis: storm releases at the clear threshold, holds on missing data
# --------------------------------------------------------------------------- #


def test_storm_holds_inside_hysteresis_band() -> None:
    """Kp 4.5 is below enter (5.0) but above clear (4.0): latch holds."""
    engine = _latched("geomagnetic-storm", RuleContext(snapshot=_snapshot(kp=6.0)))
    ctx = RuleContext(snapshot=_snapshot(kp=4.5))
    assert _outcome(engine, "geomagnetic-storm", ctx).state != "cleared"


def test_storm_clears_at_clear_threshold() -> None:
    """Kp at or below the clear threshold releases the latch with an all-clear."""
    engine = _latched("geomagnetic-storm", RuleContext(snapshot=_snapshot(kp=6.0)))
    ctx = RuleContext(snapshot=_snapshot(kp=3.5))
    outcome = _outcome(engine, "geomagnetic-storm", ctx)
    assert outcome.state == "cleared"
    assert outcome.alert is not None


def test_missing_data_holds_the_latch() -> None:
    """A data gap is not recovery: the latch survives a None Kp."""
    engine = _latched("geomagnetic-storm", RuleContext(snapshot=_snapshot(kp=6.0)))
    ctx = RuleContext(snapshot=_snapshot(kp=None))
    assert _outcome(engine, "geomagnetic-storm", ctx).state != "cleared"


# --------------------------------------------------------------------------- #
# Hysteresis: Bz release engages (previously fell through to instant clear)
# --------------------------------------------------------------------------- #


def test_bz_holds_inside_hysteresis_band() -> None:
    """Bz -4 is above enter (-5) but below clear (-2.5): latch holds."""
    engine = _latched("southward-bz", RuleContext(snapshot=_snapshot(bz=-8.0)))
    ctx = RuleContext(snapshot=_snapshot(bz=-4.0))
    assert _outcome(engine, "southward-bz", ctx).state != "cleared"


def test_bz_clears_when_northward() -> None:
    """Bz recovered to -2.0 (at/above clear -2.5) releases the latch."""
    engine = _latched("southward-bz", RuleContext(snapshot=_snapshot(bz=-8.0)))
    ctx = RuleContext(snapshot=_snapshot(bz=-2.0))
    assert _outcome(engine, "southward-bz", ctx).state == "cleared"


# --------------------------------------------------------------------------- #
# Hysteresis: ice release follows the water fraction, not the classifier edge
# --------------------------------------------------------------------------- #


def _colony(
    *,
    colony_id: str = "test-colony",
    habitat: BreedingHabitat = BreedingHabitat.FAST_ICE,
    species: str = "Aptenodytes forsteri",
) -> Colony:
    """Build a colony record for rules that resolve a scene to a colony.

    The fast-ice rules only consider scenes belonging to a fast-ice breeder, so
    a scene with no colony record is deliberately ignored by them; a context
    that wants the ice rules to act needs one of these.
    """
    return Colony(
        colony_id=colony_id,
        name="Test Colony",
        latitude=-77.0,
        longitude=166.0,
        region="Ross Sea",
        population_estimate=1000,
        population_year=2020,
        population_source="test",
        species=species,
        breeding_habitat=habitat,
        fast_ice_ratio=0.9 if habitat is BreedingHabitat.FAST_ICE else None,
    )


def _ice_outcome(engine: RuleEngine, water: float | None) -> str:
    """Evaluate one pass with a single scene at ``water`` fraction.

    ``None`` means no scenes at all: the backend went quiet.
    """
    scenes = () if water is None else (_scene(water),)
    ctx = RuleContext(snapshot=_snapshot(), scenes=scenes, colonies=(_colony(),))
    return _outcome(engine, "fast-ice-retreat", ctx).state


def _breached() -> RuleContext:
    """Return a context whose scene trips the ice enter predicate."""
    return RuleContext(snapshot=_snapshot(), scenes=(_scene(0.20),), colonies=(_colony(),))


def test_ice_holds_while_stressed() -> None:
    """Hold the ice latch while the fraction stays above the clear line.

    Fraction 0.10 reads 'stressed' (predicate false) but is above the 0.05
    clear line, so the latch holds instead of chattering at the band edge.
    """
    engine = _latched("fast-ice-retreat", _breached())
    assert _ice_outcome(engine, 0.10) != "cleared"


def test_ice_clears_when_consolidated() -> None:
    """Fraction back to 0.04 ('nominal') releases the latch."""
    engine = _latched("fast-ice-retreat", _breached())
    assert _ice_outcome(engine, 0.04) == "cleared"


def test_ice_holds_when_scenes_vanish() -> None:
    """No scenes is no evidence, not an all clear."""
    engine = _latched("fast-ice-retreat", _breached())
    assert _ice_outcome(engine, None) != "cleared"


def test_rule_without_clear_value_releases_immediately() -> None:
    """Rules with no scalar hysteresis keep the old enter-edge release."""
    rule = RuleDefinition(
        rule_id="custom",
        name="Custom",
        severity=Severity.INFO,
        predicate=lambda ctx: False,
        describe=lambda ctx: ("t", "b", {}),
        enter_threshold=1.0,
        clear_threshold=0.5,
        confirm_samples=1,
    )
    engine = RuleEngine([rule], store=None)
    assert engine._should_clear(rule, RuleContext(snapshot=_snapshot())) is True


# --------------------------------------------------------------------------- #
# net: deflate fallback raises SourceUnavailableError, never raw zlib.error
# --------------------------------------------------------------------------- #


def test_corrupt_deflate_raises_source_unavailable() -> None:
    """A body that fails both raw-deflate and zlib decoding is a source error."""
    import zlib

    response = SimpleNamespace(
        headers={"Content-Encoding": "deflate"},
        read=lambda cap: b"this is not deflate-compressed data at all" * 2,
    )
    with pytest.raises(SourceUnavailableError, match="deflate"):
        HttpClient._read_capped(response, 65536, source="test", url="https://example.invalid/x")
    # Sanity: without the fix this escapes as zlib.error, not SourceUnavailableError.
    assert not issubclass(SourceUnavailableError, zlib.error)


def test_verify_tls_false_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Disabling TLS verification leaves a trace in the logs, as documented."""
    with caplog.at_level(logging.WARNING, logger="emperor.net"):
        HttpClient(verify_tls=False)
    assert any("TLS verification disabled" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# dashboard: colony dot layout is deterministic across processes
# --------------------------------------------------------------------------- #


def test_colony_seed_is_deterministic() -> None:
    """The seed does not depend on the per-process hash() salt."""
    from emperor_space_tracker.dashboard.app import _colony_seed

    assert _colony_seed("cpe-cape-washington") == _colony_seed("cpe-cape-washington")
    assert 0 <= _colony_seed("cpe-cape-washington") < 2**31
    assert _colony_seed("colony-a") != _colony_seed("colony-b")


# --------------------------------------------------------------------------- #
# sar: footprint derives from radius_meters; Arctic branch is gone
# --------------------------------------------------------------------------- #


def test_polar_crs_rejects_non_antarctic_latitudes() -> None:
    """The wrong-hemisphere grid fails loudly instead of corrupting sigma0."""
    assert SarClient._polar_crs(-77.85) == "EPSG:3271"
    with pytest.raises(ValueError, match="outside the Antarctic domain"):
        SarClient._polar_crs(78.0)


def test_grid_geometry_uses_radius_meters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The footprint half-extent matches radius_meters, not grid*resolution."""
    from emperor_space_tracker.config import SarConfig
    from emperor_space_tracker.net import HttpClient

    captured: dict[str, object] = {}

    class _Rectangle:
        def __init__(self, lower: list[float], upper: list[float], **kwargs: object) -> None:
            captured["lower"] = lower
            captured["upper"] = upper

    fake_ee = types.ModuleType("ee")
    fake_geometry = types.SimpleNamespace(Rectangle=_Rectangle)
    fake_ee.Geometry = fake_geometry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    config = SarConfig(radius_meters=5000, grid_cells=41, resolution_meters=100)
    client = SarClient(HttpClient(), config)
    colony = _colony(colony_id="test")
    client._grid_geometry(colony)
    lower = captured["lower"]
    upper = captured["upper"]
    assert isinstance(lower, list) and isinstance(upper, list)
    # 5000 m / 111.32 km-per-degree ≈ 0.0449 degrees latitude half-extent.
    expected = 5.0 / 111.32
    assert lower[1] == pytest.approx(-77.0 - expected)
    assert upper[1] == pytest.approx(-77.0 + expected)
    # And NOT the old grid*resolution/2 = 2050 m extent.
    assert lower[1] != pytest.approx(-77.0 - 2.05 / 111.32)


# --------------------------------------------------------------------------- #
# polynya-widening: the change detector, as distinct from the classification
# bands the fast-ice-retreat rule applies to a single acquisition.
# --------------------------------------------------------------------------- #


def _trend_scenes(means: list[float], colony_id: str = "cpe-test") -> list[SarScene]:
    """Build one scene per mean-dB value, oldest first, one hour apart.

    ``mean_db`` is the only thing the trend detector reads, so the window is
    fully controlled by the list passed in.
    """
    base = utc_now() - timedelta(hours=len(means))
    return [
        SarScene(
            scene_id=f"{colony_id}-{index}",
            observed_at=base + timedelta(hours=index),
            platform="SENTINEL-1",
            orbit_type="ASCENDING",
            polarisation="VV",
            incidence_angle_deg=35.0,
            colony_id=colony_id,
            mean_db=mean,
            min_db=mean - 4.0,
            max_db=mean + 4.0,
            std_db=1.0,
            # Keep every frame "stable" so the absolute-band rule cannot fire
            # and mask whether the trend rule works on its own.
            frozen_ice_fraction=1.0,
            open_water_fraction=0.0,
            cells=(),
            provenance="synthetic.sar-simulator/v1",
        )
        for index, mean in enumerate(means)
    ]


def _polynya_ctx(
    means: list[float], habitat: BreedingHabitat = BreedingHabitat.FAST_ICE
) -> RuleContext:
    """Build a context holding a declining (or not) backscatter series.

    The colony and the scenes must agree on ``colony_id``: the trend detector
    looks a colony's series up by id, so a mismatch yields an empty series and
    an "idle" verdict that looks like a working rule rather than a broken test.
    """
    colony = _colony(colony_id="cpe-test", habitat=habitat)
    return RuleContext(
        snapshot=_snapshot(),
        scenes=tuple(_trend_scenes(means, colony_id=colony.colony_id)),
        colonies=(colony,),
    )


def _polynya_outcome(engine: RuleEngine, ctx: RuleContext) -> RuleOutcome:
    """Evaluate one pass and return the polynya outcome."""
    (found,) = [o for o in engine.evaluate(ctx) if o.rule_id == "polynya-widening"]
    return found


def test_polynya_rule_is_in_the_default_set() -> None:
    """The sustained-decline detector is actually wired up.

    ``RuleContext.trending_down`` existed and was documented for the lifetime of
    the project with no rule calling it, while the README claimed the thresholds
    were "adequate for detecting change between consecutive passes, which is what
    the trend analysis does". There was no trend analysis.
    """
    ids = {r.rule_id for r in build_default_rules(SpaceWeatherConfig())}
    assert "polynya-widening" in ids


def test_polynya_fires_on_a_sustained_decline() -> None:
    """A monotone fall of more than the threshold fires the alert."""
    engine = RuleEngine(_rules(), store=None)
    outcome = _polynya_outcome(engine, _polynya_ctx([-6.0, -6.5, -7.0, -7.6]))
    assert outcome.state == "fired"
    assert outcome.alert is not None
    assert outcome.alert.severity is Severity.SEVERE


def test_polynya_holds_but_does_not_fire_on_a_single_excursion() -> None:
    """One low acquisition among stable ones is wind roughening, not a polynya.

    The window exists to reject exactly this, and the end-to-end fall of 1.0 dB
    is under the 1.5 dB threshold anyway -- but a 2 dB single dip must still not
    fire, which is what the monotonicity half of the test enforces.
    """
    engine = RuleEngine(_rules(), store=None)
    outcome = _polynya_outcome(engine, _polynya_ctx([-6.0, -6.0, -8.0, -6.0]))
    assert outcome.state != "fired"


def test_polynya_ignores_a_rise() -> None:
    """Consolidating ice must not page anyone."""
    engine = RuleEngine(_rules(), store=None)
    outcome = _polynya_outcome(engine, _polynya_ctx([-9.0, -8.0, -7.0, -6.0]))
    assert outcome.state == "idle"


def test_polynya_needs_a_full_window() -> None:
    """Too few acquisitions is not a trend.

    Without this the detector would fire on the very first two scenes of a
    fresh node, comparing a pair of points that are 12 hours apart by accident of
    orbit geometry.
    """
    engine = RuleEngine(_rules(), store=None)
    assert _polynya_outcome(engine, _polynya_ctx([-6.0, -7.0, -8.0])).state == "idle"


def test_polynya_ignores_a_land_nesting_colony() -> None:
    """A declining series over a beach is not a polynya.

    The bands are calibrated for consolidated sea ice; the SAR client declines
    to produce such a scene, and this is the rule-side backstop.
    """
    engine = RuleEngine(_rules(), store=None)
    ctx = _polynya_ctx([-6.0, -7.0, -8.0, -9.0], habitat=BreedingHabitat.LAND)
    assert _polynya_outcome(engine, ctx).state == "idle"


def test_polynya_latch_holds_inside_the_hysteresis_band() -> None:
    """A partial recovery keeps the latch rather than chattering.

    The trend is still 1.0 dB down, which is inside the [-1.5, -0.5] band. The
    absolute-band rule would have gone idle here and the operator would get a
    "recovered" message while the ice is still clearly degrading.
    """
    engine = _latched("polynya-widening", _polynya_ctx([-6.0, -6.5, -7.0, -7.6]))

    inside = _polynya_outcome(engine, _polynya_ctx([-7.0, -7.2, -7.4, -7.6]))
    assert inside.state != "cleared", "1.0 dB is inside the hysteresis band"

    recovered = _polynya_outcome(engine, _polynya_ctx([-6.0, -6.1, -6.2, -6.3]))
    assert recovered.state == "cleared"
    assert recovered.alert is not None
    assert "recovered" in recovered.alert.body.lower()


def test_polynya_latch_holds_when_the_window_disappears() -> None:
    """Losing the acquisition history must not read as a recovery.

    This is the reason `trend_drop` returns None rather than 0.0: a store prune
    that drops old scenes leaves a short series, and a short series must not
    release a latch and emit an all clear.
    """
    engine = _latched("polynya-widening", _polynya_ctx([-6.0, -6.5, -7.0, -7.6]))
    vanished = _polynya_outcome(engine, _polynya_ctx([-6.0, -6.5]))
    assert vanished.state != "cleared"
    assert vanished.alert is None


def test_polynya_alert_reports_the_series_it_judged() -> None:
    """The message shows the window, not just the verdict.

    An operator reading a single "polynya widening" alert cannot tell how steep
    the decline is or over what period; the whole judgement is about the slope.
    """
    engine = RuleEngine(_rules(), store=None)
    outcome = _polynya_outcome(engine, _polynya_ctx([-6.0, -6.5, -7.0, -7.6]))
    assert outcome.alert is not None
    fields = dict(outcome.alert.fields)
    assert fields["drop_db"] == "-1.60"  # alert fields render floats to 2dp
    assert fields["window_scenes"] == "4"
    # The prose carries the readable slope; the field carries the machine value.
    assert "-6.00 -> -6.50 -> -7.00 -> -7.60" in outcome.alert.body
    ctx_fields = outcome.alert.context
    assert len(ctx_fields["mean_db_series"]) == 4
    assert ctx_fields["mean_db_series"][0] == -6.0
    assert ctx_fields["mean_db_series"][-1] == -7.6
    assert ctx_fields["species"] == "Aptenodytes forsteri"


def test_polynya_thresholds_come_from_config() -> None:
    """No polynya threshold is a literal in the rule code.

    The fast-ice rule's bands are still hardcoded, but a new rule must not
    repeat that mistake: these are the numbers a domain expert tunes.
    """
    config = SarConfig(
        polynya_window_scenes=6,
        polynya_min_drop_db=3.0,
        polynya_clear_drop_db=1.0,
    )
    (rule,) = [r for r in build_default_rules(SpaceWeatherConfig(), sar=config)
               if r.rule_id == "polynya-widening"]
    assert rule.enter_threshold == -3.0
    assert rule.clear_threshold == -1.0

    engine = RuleEngine([rule], store=None)
    # -2.5 dB over six scenes: past the 1.5 default, short of the 3.0 configured.
    shallow = _polynya_ctx([-6.0, -6.5, -7.0, -7.5, -8.0, -8.5])
    assert _polynya_outcome(engine, shallow).state == "idle"


def test_polynya_band_must_be_a_real_band() -> None:
    """Clear at or below enter would chatter at the threshold it detects."""
    from emperor_space_tracker.config import Config
    from emperor_space_tracker.errors import ConfigError

    config = Config(sar=SarConfig(polynya_min_drop_db=1.0, polynya_clear_drop_db=1.0))
    with pytest.raises(ConfigError, match="anti-flapping"):
        config.validate()
