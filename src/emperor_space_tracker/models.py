"""Typed observation records shared by the sources, the store and the frontend.

These are plain :mod:`dataclasses` rather than a validation library on purpose:
the daemon must not pay an import cost for schema enforcement, and the payloads
are produced by three upstream APIs whose shapes are already fixed. Parsing is
therefore *lenient at the edge and strict at the core*: helpers in
:mod:`emperor_space_tracker.sources` coerce ``None`` sentinels to ``None``
instead of guessing, and the store writes only the columns it is sure of.

Every record carries a ``provenance`` string. The dashboard renders it as a
badge so that a synthetic fast-ice backscatter sample is never mistaken for a
Genuine Sentinel-1 retrieval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "BreedingHabitat",
    "Colony",
    "FastIceCell",
    "InterplanetaryMagneticField",
    "KpIndex",
    "ObservationWindow",
    "SarScene",
    "Severity",
    "SolarWindPlasma",
    "SourceHealth",
    "SpaceWeatherSnapshot",
    "geo_distance_m",
    "utc_now",
]


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class Severity(StrEnum):
    """Alert severity ladder, ordered from least to most urgent.

    The ordering is relied upon by the rule engine when a
    ``min_severity`` floor is configured.
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    SEVERE = "severe"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Return a monotonic integer rank for comparisons."""
        return _SEVERITY_ORDER[self]

    @property
    def discord_colour(self) -> int:
        """Return the Discord embed side-bar colour for this severity."""
        return _SEVERITY_COLOUR[self]


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.DEBUG: 0,
    Severity.INFO: 1,
    Severity.WARNING: 2,
    Severity.SEVERE: 3,
    Severity.CRITICAL: 4,
}

_SEVERITY_COLOUR: dict[Severity, int] = {
    Severity.DEBUG: 0x9BA5B4,
    Severity.INFO: 0x3498DB,
    Severity.WARNING: 0xF1C40F,
    Severity.SEVERE: 0xE67E22,
    Severity.CRITICAL: 0xE74C3C,
}


@dataclass(frozen=True, slots=True)
class ObservationWindow:
    """Half-open time window ``[start, end]`` covered by a batch of readings.

    Sources that only expose a rolling window (NOAA RTSW ships the trailing day)
    report the window they actually saw so the store can be explicit about
    coverage rather than implying a single-instant snapshot.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """Reject a window whose end precedes its start.

        Without this, a reversed window makes every duration computed from
        it negative, and the result is quietly wrong rather than loud.
        """
        if self.start > self.end:
            msg = f"observation window start {self.start} is after end {self.end}"
            raise ValueError(msg)

    @property
    def seconds(self) -> float:
        """Return the window duration in seconds."""
        return (self.end - self.start).total_seconds()

    def contains(self, moment: datetime) -> bool:
        """Return whether ``moment`` falls inside the window."""
        return self.start <= moment <= self.end


@dataclass(frozen=True, slots=True)
class SolarWindPlasma:
    """A single real-time solar wind plasma sample from the DSCOVR/RTSW feed.

    SWPC's RTSW endpoint multiplexes DSCOVR, ACE and IMAP onto one stream and
    tags each record with ``source`` and ``active``. Only ``active`` records
    describe the currently published spacecraft, so
    :meth:`is_nominal` is the guard before any value is used for alerting.
    """

    observed_at: datetime
    source: str
    speed_kms: float | None
    density_per_cm3: float | None
    temperature_k: float | None
    active: bool
    quality: int | None = None
    provenance: str = "noaa.swpc.rtsw"

    def is_nominal(self) -> bool:
        """Return whether the record is usable for threshold evaluation.

        SWPC encodes "no data" as ``9999`` in the text products; the JSON feeds
        use ``null`` but spacecraft faults still surface as out-of-family
        magnitudes, so both are screened.
        """
        if not self.active:
            return False
        if self.speed_kms is None:
            return False
        # Plausible solar wind at 1 AU: 100 km/s (slow wind) to 1200 km/s.
        # ICMEs and the fastest coronal compressions sit near the top of this.
        return 100.0 <= self.speed_kms <= 1500.0


@dataclass(frozen=True, slots=True)
class InterplanetaryMagneticField:
    """A single magnetic field sample from the DSCOVR/RTSW feed.

    ``bz_gsm`` is the component that matters: a sustained southward Bz couples to
    the magnetopause and drives geomagnetic activity, which in turn expands the
    auroral oval poleward and degrades HF radio and satellite drag at high
    latitude.
    """

    observed_at: datetime
    source: str
    bt_nt: float | None
    bz_gsm_nt: float | None
    by_gsm_nt: float | None
    density_pct: float | None
    active: bool
    provenance: str = "noaa.swpc.rtsw"

    def is_nominal(self) -> bool:
        """Return whether the field vector is usable for threshold evaluation."""
        if not self.active or self.bz_gsm_nt is None:
            return False
        return abs(self.bz_gsm_nt) < 100.0


@dataclass(frozen=True, slots=True)
class KpIndex:
    """A planetary Kp index estimate at three-hour cadence.

    ``estimated_kp`` is the nowcast derived by SWPC from magnetometer station
    data and updates every minute; ``kp_index`` is the official three-hour
    barometer value and is often ``None`` for the current, incomplete barometer
    interval. The engine prefers the estimate and records which it used.
    """

    observed_at: datetime
    kp_index: int | None
    estimated_kp: float | None
    provenance: str = "noaa.swpc.planetary_k_index"

    @property
    def value(self) -> float | None:
        """Return the best available Kp value, preferring the nowcast."""
        if self.estimated_kp is not None:
            return float(self.estimated_kp)
        if self.kp_index is not None:
            return float(self.kp_index)
        return None

    @property
    def is_estimated(self) -> bool:
        """Return whether :attr:`value` came from the SWPC nowcast."""
        return self.estimated_kp is not None

    def storm_level(self, threshold: float) -> bool:
        """Return whether Kp is at or above ``threshold`` (a G-scale storm)."""
        value = self.value
        return value is not None and value >= threshold


@dataclass(frozen=True, slots=True)
class SpaceWeatherSnapshot:
    """Everything the rule engine needs to judge one polling instant.

    Assembled by :meth:`emperor_space_tracker.sources.space_weather.build_snapshot`
    from independently-fetched SWPC products. Each field is optional because the
    feeds fail independently; a snapshot with a ``None`` plasma speed is still
    useful for Kp-based alerting.
    """

    observed_at: datetime
    plasma: SolarWindPlasma | None = None
    magnetic_field: InterplanetaryMagneticField | None = None
    kp: KpIndex | None = None
    window: ObservationWindow | None = None
    f107_sfu: float | None = None
    degraded: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_degraded(self) -> bool:
        """Return whether one or more upstream products failed to load."""
        return bool(self.degraded)

    def headline(self) -> str:
        """Return a one-line human summary for logs and alert embeds."""
        bits: list[str] = []
        if self.plasma is not None and self.plasma.speed_kms is not None:
            bits.append(f"{self.plasma.speed_kms:.0f} km/s")
        if self.magnetic_field is not None and self.magnetic_field.bz_gsm_nt is not None:
            bits.append(f"Bz {self.magnetic_field.bz_gsm_nt:+.1f} nT")
        kp = self.kp.value if self.kp is not None else None
        if kp is not None:
            bits.append(f"Kp {kp:.1f}")
        if not bits:
            return "no space weather data available"
        return " · ".join(bits)


@dataclass(frozen=True, slots=True)
class SarScene:
    """A Sentinel-1 GRD acquisition reduced to a colony-centred backscatter grid.

    ``polarisation`` names the band each ``sigma0_db`` value came from, so the
    same schema serves VV and HH without a migration.
    Fast ice reads bright (roughly -8 to -2 dB) because brine channels and
    pressure ridges act as volume scatterers, while open water and thin nilas
    read dark (roughly -20 to -12 dB). A *drop* in sigma0 between acquisitions
    is the standard remote-sensing signature of a lead opening or the fast ice
    detaching from the coast. The band is whichever the scene records in
    ``polarisation``; the numbers below are indicative for C-band.
    """

    scene_id: str
    observed_at: datetime
    platform: str
    orbit_type: str
    polarisation: str
    incidence_angle_deg: float | None
    colony_id: str
    mean_db: float
    min_db: float
    max_db: float
    std_db: float
    frozen_ice_fraction: float
    open_water_fraction: float
    cells: tuple[FastIceCell, ...]
    provenance: str
    bytes_processed: int = 0

    @property
    def is_synthetic(self) -> bool:
        """Return whether the scene was simulated rather than retrieved."""
        return self.provenance.startswith("synthetic")

    @property
    def open_water_fraction_percent(self) -> float:
        """Return the open-water share of the grid as a percentage."""
        return self.open_water_fraction * 100.0

    def classify(self) -> str:
        """Classify fast-ice stability from the open-water fraction.

        The banding is deliberately coarse: a 100 m Sentinel-1 grid cell at
        1 km posting resolves leads and polynyas, not fine fracture, so the
        output is a stability grade rather than a continuous risk index.

        Returns
        -------
        str
            One of ``stable``, ``nominal``, ``stressed``, ``breached`` or
            ``dispersed``.
        """
        water = self.open_water_fraction
        if water < 0.01:
            return "stable"
        if water < 0.05:
            return "nominal"
        if water < 0.15:
            return "stressed"
        if water < 0.40:
            return "breached"
        return "dispersed"


@dataclass(frozen=True, slots=True)
class FastIceCell:
    """One grid cell of a :class:`SarScene` backscatter matrix.

    Stored as a row in the ``sar_cells`` table and rendered as the heatmap axes
    of the dashboard's radar tab.
    """

    scene_id: str
    row: int
    col: int
    sigma0_db: float
    is_open_water: bool
    classification: str

    def as_matrix_value(self) -> float:
        """Return the plotting value, using a sentinel for open water.

        Plotly renders ``None`` as a transparent gap, which is the honest
        representation of "open water, no ice surface to scatter from".
        """
        return math.nan if self.is_open_water else self.sigma0_db


class BreedingHabitat(StrEnum):
    """Where a penguin species builds its nest.

    This is the discriminator that decides which of this node's monitoring
    channels are valid for a colony, and it is deliberately a statement about
    nesting substrate rather than about latitude. A species' range and its
    breeding substrate are independent: the Kerguelen gentoo
    (*Pygoscelis kerguelensis*) breeds at 49 degrees south, on bare ground,
    with no polar night at all.

    Attributes
    ----------
    FAST_ICE
        The nest is on sea ice that the colony must access. Only the Emperor
        penguin does this. Fast-ice SAR backscatter is a valid proxy for
        breeding viability, and the fast-ice alert applies.
    LAND
        The nest is on shore -- beach, moraine, gravel, vegetation, rock. This
        is every other *Pygoscelis*, the crested auklets, the king penguins, and
        so on. Fast-ice SAR backscatter carries no information about whether
        such a colony is doing well, so the fast-ice alert and the SAR scene
        pipeline are skipped rather than being allowed to emit a reading that
        looks plausible and means nothing.
    """

    FAST_ICE = "fast_ice"
    LAND = "land"

    @property
    def uses_fast_ice(self) -> bool:
        """Return whether SAR fast-ice monitoring is valid for this habitat."""
        return self is BreedingHabitat.FAST_ICE


@dataclass(frozen=True, slots=True)
class Colony:
    """A penguin breeding location with census metadata.

    Population figures are per-fledging or per-breeding-pair estimates drawn
    from published censuses, not from GBIF occurrence density.
    :attr:`population_source` keeps those two provenance streams
    distinguishable.

    A colony belongs to exactly one species, named by :attr:`species`, and that
    species' :attr:`breeding_habitat` decides which of this node's monitoring
    channels apply. The two are not interchangeable: a fast-ice breeder is
    observable with SAR backscatter during polar night, and a land-nesting
    breeder is not observable with it at all.

    Attributes
    ----------
    species
        Binomial scientific name, e.g. ``"Aptenodytes forsteri"``.
    breeding_habitat
        Where the species builds its nest. This is the gate on the SAR and
        fast-ice alerting, so it is a required field rather than an inferred
        one.
    fast_ice_ratio
        Proportion of the fast-ice season the colony can access. Only
        meaningful for a ``FAST_ICE`` breeder; :attr:`monitors_fast_ice` is the
        single predicate the rest of the codebase should consult rather than
        testing the value for ``None`` itself.
    common_name
        Optional vernacular name, used for display only. Never a key.
    """

    colony_id: str
    name: str
    latitude: float
    longitude: float
    region: str
    population_estimate: int | None
    population_year: int | None
    population_source: str
    species: str
    breeding_habitat: BreedingHabitat
    fast_ice_ratio: float | None = None
    last_census_at: datetime | None = None
    notes: str = ""
    common_name: str = ""

    @property
    def monitors_fast_ice(self) -> bool:
        """Return whether fast-ice SAR monitoring applies to this colony.

        The single predicate that decides if a SAR scene and the fast-ice
        alert mean anything for this colony. Checking :attr:`breeding_habitat`
        here rather than testing :attr:`fast_ice_ratio` for ``None`` at each
        call site keeps that decision in one place, so a species whose census
        happens to lack an ice figure is not silently promoted to being
        ice-observable.
        """
        return self.breeding_habitat is BreedingHabitat.FAST_ICE

    @property
    def is_major(self) -> bool:
        """Return whether the colony holds at least 10,000 breeding pairs."""
        return (self.population_estimate or 0) >= 10_000

    @property
    def label(self) -> str:
        """Return a short display label suitable for map annotation."""
        population = f"{self.population_estimate:,}" if self.population_estimate else "n/a"
        return f"{self.name} ({population})"

    def as_geojson(self) -> dict[str, Any]:
        """Return a GeoJSON ``Feature`` for the dashboard's map layers."""
        properties: dict[str, Any] = {
            "colony_id": self.colony_id,
            "name": self.name,
            "region": self.region,
            "species": self.species,
            "common_name": self.common_name,
            "breeding_habitat": str(self.breeding_habitat),
            "monitors_fast_ice": self.monitors_fast_ice,
            "population_estimate": self.population_estimate,
            "population_year": self.population_year,
            "population_source": self.population_source,
            "fast_ice_ratio": self.fast_ice_ratio,
            "notes": self.notes,
        }
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [self.longitude, self.latitude]},
            "properties": properties,
        }


@dataclass(frozen=True, slots=True)
class SourceHealth:
    """Per-source outcome of a single polling pass.

    Recorded so the dashboard can distinguish "quiet space weather" from
    "the feed is broken" — a distinction that matters enormously during polar
    night when a human cannot go and look.
    """

    source: str
    ok: bool
    latency_ms: int
    detail: str = ""
    record_count: int = 0
    observed_at: datetime = field(default_factory=utc_now)

    @property
    def status(self) -> str:
        """Return ``ok``, ``degraded`` or ``down`` for display."""
        if not self.ok:
            return "down"
        return "degraded" if self.record_count == 0 else "ok"


def geo_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two coordinates.

    Uses the haversine formula on a spherical Earth. Over the scale of a single
    fast-ice tile (< 100 km) the spherical error is under 0.5 %, which is well
    inside the uncertainty of Sentinel-1 geolocation at 100 m posting.

    Parameters
    ----------
    lat1, lon1, lat2, lon2
        Latitude and longitude in decimal degrees.

    Returns
    -------
    float
        Distance in metres.

    Examples
    --------
    >>> round(geo_distance_m(-77.85, 166.67, -77.85, 166.67))
    0
    >>> 20000 < geo_distance_m(-77.85, 166.67, -78.03, 166.67) < 21000
    True
    """
    radius_m = 6_371_008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(min(1.0, a)))
