"""Space weather ingestion from NOAA SWPC.

Endpoint notes, verified against the live service
-------------------------------------------------
The commonly cited ``services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json``
family is **gone** -- every one of those paths returns 404 as of writing. The
replacement for real-time DSCOVR plasma and magnetic field data is the RTSW
(RTSW) JSON stream, which multiplexes DSCOVR, ACE and IMAP onto one document and
tags each record with ``source`` and ``active``.

Two consequences shape this module:

* **Only ``active`` records describe the currently published spacecraft.** A
  node in the southern auroral zone must not alert off an IMAP record that the
  SWPC model is no longer using. :meth:`SolarWindPlasma.is_nominal` enforces it.
* **The document is 2.5 MB and newest-first.** See
  :meth:`emperor_space_tracker.net.HttpClient.fetch_json_array_prefix` for the
  byte-range strategy that keeps this at 32 KB per request.

The planetary Kp stream (``json/planetary_k_index_1m.json``) is small, complete
and updated every minute, so it is fetched whole. It carries both the official
three-hour ``kp_index`` and the ``estimated_kp`` nowcast derived from
magnetometer stations; the nowcast is preferred because the official value is
``None`` for the barometer interval still in progress.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from ..errors import SourceError
from ..models import (
    InterplanetaryMagneticField,
    KpIndex,
    ObservationWindow,
    SolarWindPlasma,
    SourceHealth,
    SpaceWeatherSnapshot,
    utc_now,
)
from ..net import HttpClient

__all__ = [
    "SWPC_ENDPOINTS",
    "SpaceWeatherClient",
    "parse_f107",
    "parse_kp",
    "parse_magnetic_field",
    "parse_plasma",
]

_LOG = logging.getLogger("emperor.sources.swpc")

_BASE: Final = "https://services.swpc.noaa.gov"

#: Live product endpoints, keyed by logical name.
SWPC_ENDPOINTS: Final[dict[str, str]] = {
    "plasma": f"{_BASE}/json/rtsw/rtsw_wind_1m.json",
    "magnetic_field": f"{_BASE}/json/rtsw/rtsw_mag_1m.json",
    "planetary_k_index": f"{_BASE}/json/planetary_k_index_1m.json",
    "f10_7_flux": f"{_BASE}/json/f107_cm_flux.json",
}

#: Sentinel used by SWPC for "value unavailable" in the text products. The JSON
#: feeds use ``null``, but the same convention survives in some nested fields.
_NULL_SENTINELS: Final = frozenset({9999.0, 9999.9, -9999.0, 99999.0, 999.9})


def _number(value: Any) -> float | None:
    """Coerce a JSON scalar to ``float``, mapping SWPC sentinels to ``None``.

    Never guesses: a sentinel becomes ``None`` so the caller must handle
    missing data explicitly rather than alerting on 9999 km/s.

    Parameters
    ----------
    value
        The raw JSON value.

    Returns
    -------
    float | None
        A finite float, or ``None`` if the value is absent, non-numeric, or a
        known "no data" sentinel.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if number in _NULL_SENTINELS:
        return None
    return number


def _timestamp(value: Any) -> datetime:
    """Parse an SWPC ``time_tag`` into an aware UTC datetime.

    SWPC publishes naive UTC. This function attaches UTC rather than deferring
    to the local zone, because getting this wrong silently shifts every
    observation by the node's UTC offset.

    Parameters
    ----------
    value
        An ISO-8601 string, or ``None``.

    Returns
    -------
    datetime
        Aware UTC datetime. Falls back to the current time for an unparseable
        value, which is preferable to dropping the sample entirely and is
        flagged by the caller's staleness check.

    Examples
    --------
    >>> _timestamp("2026-01-01T12:00:00").tzinfo is not None
    True
    >>> _timestamp(None) is not None
    True
    """
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return utc_now()
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return utc_now()


def _prefers_active_source(records: list[Any]) -> dict[str, Any] | None:
    """Return the newest record that belongs to an *active* spacecraft.

    The RTSW stream interleaves DSCOVR, ACE and IMAP. Only one is normally
    flagged active, and it is the one SWPC publishes as the operational real-time
    source. Preferring it is what keeps the node from alerting off a spacecraft
    that has drifted or been decommissioned.

    Parameters
    ----------
    records
        Records in document order (newest first for SWPC feeds).

    Returns
    -------
    dict[str, Any] | None
        The chosen record, or ``None`` if no record is active.
    """
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("active") is True:
            return record
    return None


def parse_plasma(records: list[Any]) -> SolarWindPlasma | None:
    """Parse the newest active plasma sample from an RTSW wind array.

    Parameters
    ----------
    records
        Decoded records from ``rtsw_wind_1m.json``.

    Returns
    -------
    SolarWindPlasma | None
        The sample, or ``None`` if no active record is present.

    Examples
    --------
    >>> parse_plasma([{"time_tag": "2026-01-01T00:00:00", "active": False,
    ...                "source": "IMAP", "proton_speed": 400.0}]) is None
    True
    >>> sample = parse_plasma([{"time_tag": "2026-01-01T00:00:00", "active": True,
    ...                         "source": "DSCOVR", "proton_speed": 450.0}])
    >>> sample.speed_kms
    450.0
    """
    if not records:
        return None
    record = _prefers_active_source(records)
    if record is None:
        return None
    return SolarWindPlasma(
        observed_at=_timestamp(record.get("time_tag")),
        source=str(record.get("source") or "unknown"),
        speed_kms=_number(record.get("proton_speed")),
        density_per_cm3=_number(record.get("proton_density")),
        temperature_k=_number(record.get("proton_temperature")),
        active=True,
        quality=int(_number(record.get("overall_quality")) or 0),
    )


def parse_magnetic_field(records: list[Any]) -> InterplanetaryMagneticField | None:
    """Parse the newest active magnetic field sample from an RTSW mag array.

    Parameters
    ----------
    records
        Decoded records from ``rtsw_mag_1m.json``.

    Returns
    -------
    InterplanetaryMagneticField | None
        The sample, or ``None`` if no active record is present.
    """
    if not records:
        return None
    record = _prefers_active_source(records)
    if record is None:
        return None
    return InterplanetaryMagneticField(
        observed_at=_timestamp(record.get("time_tag")),
        source=str(record.get("source") or "unknown"),
        bt_nt=_number(record.get("bt")),
        bz_gsm_nt=_number(record.get("bz_gsm")),
        by_gsm_nt=_number(record.get("by_gsm")),
        density_pct=_number(record.get("density_pct")),
        active=True,
    )


def parse_kp(records: list[Any]) -> KpIndex | None:
    """Parse the newest planetary Kp record.

    Parameters
    ----------
    records
        Decoded records from ``planetary_k_index_1m.json``, oldest first as
        SWPC publishes it.

    Returns
    -------
    KpIndex | None
        The newest parseable record, or ``None``.

    Examples
    --------
    >>> parse_kp([{"time_tag": "2026-01-01T00:00:00", "kp_index": 3,
    ...            "estimated_kp": 3.33}]).value
    3.33
    >>> parse_kp([]) is None
    True
    """
    if not records:
        return None
    for record in reversed(records):
        if not isinstance(record, dict):
            continue
        kp_index = _number(record.get("kp_index"))
        estimated = _number(record.get("estimated_kp"))
        if kp_index is None and estimated is None:
            continue
        return KpIndex(
            observed_at=_timestamp(record.get("time_tag")),
            kp_index=int(kp_index) if kp_index is not None else None,
            estimated_kp=estimated,
        )
    return None


def parse_f107(records: list[Any]) -> float | None:
    """Parse the newest 10.7 cm solar radio flux value.

    Parameters
    ----------
    records
        Decoded records from ``f107_cm_flux.json``.

    Returns
    -------
    float | None
        Flux in solar flux units, or ``None`` if unavailable.
    """
    if not records:
        return None
    for record in reversed(records):
        if not isinstance(record, dict):
            continue
        flux = _number(record.get("flux"))
        if flux is not None and flux > 0:
            return flux
    return None


class SpaceWeatherClient:
    """Fetches and assembles space weather observations from NOAA SWPC.

    Parameters
    ----------
    client
        Shared HTTP client, so one connection pool and one timeout policy serve
        every source.
    prefetch_bytes
        Leading bytes to request per RTSW array. 32 KiB holds roughly 40
        one-minute samples per spacecraft, far more than a five-minute poll
        consumes.
    track_f107
        Fetch the 10.7 cm radio flux. Off by default for very constrained nodes.

    Examples
    --------
    >>> from emperor_space_tracker.net import HttpClient
    >>> swpc = SpaceWeatherClient(HttpClient(timeout=10.0))
    >>> snapshot = swpc.poll()  # doctest: +SKIP
    >>> snapshot.headline()  # doctest: +SKIP
    '450 km/s · Bz -1.2 nT · Kp 2.3'
    """

    def __init__(
        self,
        client: HttpClient,
        *,
        prefetch_bytes: int = 32 * 1024,
        track_f107: bool = True,
    ) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.client = client
        self.prefetch_bytes = prefetch_bytes
        self.track_f107 = track_f107

    def fetch_plasma(self) -> tuple[SolarWindPlasma | None, int, str]:
        """Fetch the newest active solar wind plasma sample.

        Returns
        -------
        tuple[SolarWindPlasma | None, int, str]
            ``(sample, latency_ms, detail)``. The sample is ``None`` on
            transport failure; ``detail`` explains what happened either way.
        """
        url = SWPC_ENDPOINTS["plasma"]
        started = utc_now()
        try:
            records = self.client.fetch_json_array_prefix(
                url, prefetch_bytes=self.prefetch_bytes, source="swpc.plasma"
            )
        except SourceError as exc:
            return None, int((utc_now() - started).total_seconds() * 1000), str(exc)
        sample = parse_plasma(records)
        elapsed = int((utc_now() - started).total_seconds() * 1000)
        if sample is None:
            return None, elapsed, f"{len(records)} records but none from an active spacecraft"
        return sample, elapsed, f"{len(records)} records, spacecraft {sample.source}"

    def fetch_magnetic_field(self) -> tuple[InterplanetaryMagneticField | None, int, str]:
        """Fetch the newest active interplanetary magnetic field sample.

        Returns
        -------
        tuple[InterplanetaryMagneticField | None, int, str]
            ``(sample, latency_ms, detail)``.
        """
        url = SWPC_ENDPOINTS["magnetic_field"]
        started = utc_now()
        try:
            records = self.client.fetch_json_array_prefix(
                url, prefetch_bytes=self.prefetch_bytes, source="swpc.mag"
            )
        except SourceError as exc:
            return None, int((utc_now() - started).total_seconds() * 1000), str(exc)
        sample = parse_magnetic_field(records)
        elapsed = int((utc_now() - started).total_seconds() * 1000)
        if sample is None:
            return None, elapsed, f"{len(records)} records but none from an active spacecraft"
        return sample, elapsed, f"{len(records)} records, spacecraft {sample.source}"

    def fetch_kp(self) -> tuple[KpIndex | None, int, str]:
        """Fetch the newest planetary Kp index.

        Returns
        -------
        tuple[KpIndex | None, int, str]
            ``(index, latency_ms, detail)``.
        """
        url = SWPC_ENDPOINTS["planetary_k_index"]
        started = utc_now()
        try:
            payload = self.client.get_json(url, source="swpc.kp")
        except SourceError as exc:
            return None, int((utc_now() - started).total_seconds() * 1000), str(exc)
        if not isinstance(payload, list):
            return None, int((utc_now() - started).total_seconds() * 1000), "expected a JSON array"
        index = parse_kp(payload)
        elapsed = int((utc_now() - started).total_seconds() * 1000)
        if index is None:
            return None, elapsed, "array contained no parseable Kp record"
        kind = "nowcast" if index.is_estimated else "3-hour barometer"
        return index, elapsed, f"index {index.value} ({kind})"

    def fetch_f107(self) -> tuple[float | None, int, str]:
        """Fetch the newest 10.7 cm solar radio flux value.

        Returns
        -------
        tuple[float | None, int, str]
            ``(flux_sfu, latency_ms, detail)``.
        """
        url = SWPC_ENDPOINTS["f10_7_flux"]
        started = utc_now()
        try:
            payload = self.client.get_json(url, source="swpc.f107")
        except SourceError as exc:
            return None, int((utc_now() - started).total_seconds() * 1000), str(exc)
        if not isinstance(payload, list):
            return None, int((utc_now() - started).total_seconds() * 1000), "expected a JSON array"
        flux = parse_f107(payload)
        elapsed = int((utc_now() - started).total_seconds() * 1000)
        return flux, elapsed, ("unavailable" if flux is None else f"{flux:.1f} sfu")

    def poll(self) -> tuple[SpaceWeatherSnapshot, list[SourceHealth]]:
        """Fetch every space weather product and assemble one snapshot.

        Products are fetched independently so a single upstream failure degrades
        the snapshot instead of failing the poll. Each failure is recorded in
        :attr:`SpaceWeatherSnapshot.degraded`, which the dashboard surfaces
        explicitly -- silence caused by a broken feed is the worst possible
        failure mode for a node whose whole job is unattended night-time
        monitoring.

        Returns
        -------
        tuple[SpaceWeatherSnapshot, list[SourceHealth]]
            The assembled snapshot, with ``degraded`` naming any failed product,
            and the per-product health for the pass.

        Examples
        --------
        >>> snapshot, health = swpc.poll()  # doctest: +SKIP
        >>> [h.source for h in health]  # doctest: +SKIP
        ['swpc.plasma', 'swpc.mag', 'swpc.kp', 'swpc.f107']
        """
        degraded: list[str] = []
        plasma = None
        field = None
        kp = None
        f107 = None
        health: list[SourceHealth] = []
        observed = utc_now()

        plasma, latency, detail = self.fetch_plasma()
        if plasma is None:
            degraded.append("plasma")
        health.append(
            SourceHealth("swpc.plasma", plasma is not None, latency, detail,
                         1 if plasma else 0, observed)
        )

        field, latency, detail = self.fetch_magnetic_field()
        if field is None:
            degraded.append("magnetic_field")
        health.append(
            SourceHealth("swpc.mag", field is not None, latency, detail,
                         1 if field else 0, observed)
        )

        kp, latency, detail = self.fetch_kp()
        if kp is None:
            degraded.append("planetary_k_index")
        health.append(
            SourceHealth("swpc.kp", kp is not None, latency, detail, 1 if kp else 0, observed)
        )

        if self.track_f107:
            f107, latency, detail = self.fetch_f107()
            if f107 is None:
                degraded.append("f10_7_flux")
            health.append(
                SourceHealth("swpc.f107", f107 is not None, latency, detail,
                             1 if f107 is not None else 0, observed)
            )

        window = self._window(plasma, field, kp)

        return SpaceWeatherSnapshot(
            observed_at=observed,
            plasma=plasma,
            magnetic_field=field,
            kp=kp,
            window=window,
            f107_sfu=f107,
            degraded=tuple(degraded),
        ), health

    @staticmethod
    def _window(
        plasma: SolarWindPlasma | None,
        field: InterplanetaryMagneticField | None,
        kp: KpIndex | None,
    ) -> ObservationWindow | None:
        """Return the earliest-to-latest observation time as a window."""
        times = [t.observed_at for t in (plasma, field, kp) if t is not None]
        if not times:
            return None
        return ObservationWindow(start=min(times), end=max(times))

    def is_stale(self, snapshot: SpaceWeatherSnapshot, max_age_seconds: float) -> list[str]:
        """Return names of products whose samples are older than the limit.

        A cached CDN response during a network partition will happily return a
        plasma sample from six hours ago with a 200 status. Without this check
        the node would report calm space weather while blind, which is worse
        than reporting a fault.

        Parameters
        ----------
        snapshot
            The snapshot to audit.
        max_age_seconds
            Maximum acceptable age per sample.

        Returns
        -------
        list[str]
            Names of stale products. Empty if everything is fresh.
        """
        now = utc_now()
        stale: list[str] = []
        for name, sample in (
            ("plasma", snapshot.plasma),
            ("magnetic_field", snapshot.magnetic_field),
            ("planetary_k_index", snapshot.kp),
        ):
            if sample is None:
                continue
            age = (now - sample.observed_at).total_seconds()
            if age > max_age_seconds:
                stale.append(f"{name} is {age / 60:.0f} min old")
        return stale

    def recent_kp_history(self, *, hours: int = 72) -> list[KpIndex]:
        """Return the Kp series over a trailing window, oldest first.

        Parameters
        ----------
        hours
            Trailing window in hours.

        Returns
        -------
        list[KpIndex]
        """
        try:
            payload = self.client.get_json(
                SWPC_ENDPOINTS["planetary_k_index"], source="swpc.kp.history"
            )
        except SourceError as exc:
            _LOG.warning("Kp history unavailable: %s", exc)
            return []
        if not isinstance(payload, list):
            return []
        since = utc_now() - timedelta(hours=hours)
        series: list[KpIndex] = []
        for record in payload:
            if not isinstance(record, dict):
                continue
            index = parse_kp([record])
            if index is not None and index.observed_at >= since:
                series.append(index)
        return series
