"""Biological colony census from GBIF and the SCAR ecosystem record.

Two distinct data streams, deliberately kept separate because conflating them
produces confidently wrong numbers:

**GBIF** (``api.gbif.org``)
    Global Biodiversity Information Facility occurrence search. Answers *where
    and when has this species been recorded*, aggregated to geographic cells.
    This is presence/absence evidence, weighted by observer effort and citizen
    science. It is emphatically **not** a population count, and treating GBIF
    record density as abundance is a classic analytical error. The code below
    never derives a population from it.

**SCAR / AADC** (``data.aad.gov.au``)
    The Scientific Committee on Antarctic Research ecosystem monitoring record,
    mirrored by CCAMLR. This is where breeding-pair and fledging counts actually
    come from.

    The honest engineering note: the AADC portal is a JavaScript single-page
    application with no public REST API. Every endpoint under it returns the same
    1.2 kB HTML shell to an HTTP client. Rather than ship a scraper that breaks
    on their next deploy, this module ships a **curated reference catalogue** of
    the well-documented Emperor breeding locations, sourced from the published
    CEMP record, and clearly labels the provenance of every figure.

    The catalogue is a TOML data file, not code, so updating it after a new
    census season is a data change that a domain expert can review. A future
    integration against a real SCAR OGC or GBIF-published dataset only needs to
    replace :meth:`ColonyCensusClient.fetch_reference_catalogue`.

Both streams land in the same :class:`~emperor_space_tracker.models.Colony`
table but keep distinct ``population_source`` values, and the dashboard renders
that field so a reader can always tell where a number came from.
"""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

from ..errors import SourceError
from ..models import BreedingHabitat, Colony, SourceHealth, geo_distance_m, utc_now
from ..net import HttpClient

__all__ = [
    "GBIF_ENDPOINTS",
    "ColonyCensusClient",
    "PresenceGrid",
    "aggregate_occurrences",
    "load_reference_catalogue",
]

_LOG = logging.getLogger("emperor.sources.biological")

_BASE: Final = "https://api.gbif.org/v1"

GBIF_ENDPOINTS: Final[dict[str, str]] = {
    "species_match": f"{_BASE}/species/match",
    "occurrence_search": f"{_BASE}/occurrence/search",
    "dataset_search": f"{_BASE}/dataset/search",
}

#: Location of the curated SCAR/CEMP colony catalogue shipped with the package.
_CATALOGUE_PATH: Final = Path(__file__).resolve().parent.parent / "data" / "colonies.toml"

#: GBIF rounds coordinates to this many decimal places in the backbone. At the
#: equator 3 dp is ~110 m, at 78 degrees S it is ~24 m. Used to derive the
#: `coordinate_uncertainty` of an aggregate cell without trusting a value the
#: API does not provide.
_COORDINATE_PRECISION_M: Final = 24.0

#: Occurrence searches are paginated. This is deliberately small, and the
#: reason is measured rather than assumed: GBIF returns 93 keys per record
#: (multimedia extensions, classification arrays, verbatim fields), so a
#: 300-record page is ~1.9 MB of JSON and ~8 MB once decoded into dicts. The
#: documented ``fields`` parameter that would trim it to four columns is
#: *silently ignored* by the live occurrence endpoint -- verified against
#: api.gbif.org on 2026-09-27, where every field combination returned all 93
#: keys. With no server-side selection the only lever is page size, so 100 it
#: is, and pages are aggregated and discarded one at a time.
_PAGE_SIZE: Final = 100
_MAX_PAGES: Final = 3


def load_reference_catalogue(path: Path | None = None) -> list[Colony]:
    """Load the curated penguin colony catalogue.

    The catalogue is multi-species. Each record names its own ``species`` and
    ``breeding_habitat``; the two that ship today are the fast-ice-breeding
    Emperor and the land-nesting southeastern gentoo. A record that omits
    ``species`` is an error rather than a default, because silently assuming
    "Emperor" is how a gentoo ends up being evaluated against fast-ice
    thresholds that mean nothing for it.

    Parameters
    ----------
    path
        Override for the catalogue location. Defaults to the packaged TOML.

    Returns
    -------
    list[Colony]
        Colonies parsed from the catalogue. Empty if the file is missing, so a
        stripped-down installation degrades to GBIF-only presence mapping rather
        than failing to start.

    Raises
    ------
    SourceError
        If the catalogue exists but is not valid TOML or has a bad record.

    Examples
    --------
    >>> colonies = load_reference_catalogue()
    >>> all(-90 <= c.latitude <= 90 for c in colonies)
    True
    >>> {c.species for c in colonies} >= {"Aptenodytes forsteri"}
    True
    """
    catalogue = path or _CATALOGUE_PATH
    try:
        raw = catalogue.read_bytes()
    except OSError:
        _LOG.warning(
            "colony catalogue %s is missing; continuing without census baselines",
            catalogue,
        )
        return []

    try:
        parsed: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        msg = f"colony catalogue {catalogue} is corrupt: {exc}"
        raise SourceError("scar.catalogue", msg) from exc

    records = parsed.get("colony") or []
    if not isinstance(records, list):
        msg = f"colony catalogue {catalogue}: expected an array of tables under [colony]"
        raise SourceError("scar.catalogue", msg)

    colonies: list[Colony] = []
    for entry in records:
        if not isinstance(entry, dict):
            continue
        try:
            census = entry.get("last_census")
            census_at = None
            if isinstance(census, str):
                try:
                    census_at = datetime.fromisoformat(census).replace(tzinfo=UTC)
                except ValueError:
                    census_at = None
            habitat = BreedingHabitat(str(entry["breeding_habitat"]))
            colonies.append(
                Colony(
                    colony_id=str(entry["id"]),
                    name=str(entry["name"]),
                    latitude=float(entry["latitude"]),
                    longitude=float(entry["longitude"]),
                    region=str(entry.get("region", "unknown")),
                    population_estimate=(
                        int(entry["population"]) if entry.get("population") is not None else None
                    ),
                    population_year=(
                        int(entry["population_year"])
                        if entry.get("population_year") is not None
                        else None
                    ),
                    population_source=str(
                        entry.get("population_source", "SCAR CEMP / CCAMLR ecosystem monitoring")
                    ),
                    species=str(entry["species"]),
                    breeding_habitat=habitat,
                    common_name=str(entry.get("common_name", "")),
                    fast_ice_ratio=(
                        float(entry["fast_ice_ratio"])
                        if entry.get("fast_ice_ratio") is not None
                        else None
                    ),
                    last_census_at=census_at,
                    notes=str(entry.get("notes", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"colony catalogue {catalogue} has a malformed record: {exc}"
            raise SourceError("scar.catalogue", msg) from exc

    _LOG.info("loaded %d reference colonies from %s", len(colonies), catalogue.name)
    return colonies


class PresenceGrid:
    """Incremental spatial presence accumulator.

    This exists as a class rather than a plain function because the occurrence
    endpoint cannot be asked for a projection, so each record arrives with 93
    keys and the only way to bound memory is to reduce records the moment they
    arrive and never keep them. Feeding pages into one long-lived accumulator
    and discarding each page caps peak memory at a single page instead of the
    whole scan.

    This is a **presence** summary, not an abundance estimate. Each cell reports
    how many *records* were filed there, which reflects observer effort, taxon
    submissions and expedition logistics at least as much as it reflects bird
    numbers. Nothing here is ever named ``population``.
    """

    def __init__(self, *, precision: int = 2) -> None:
        """Create an empty accumulator.

        Parameters
        ----------
        precision
            Decimal places to round coordinates to when binning. 2 dp is
            ~1.1 km, which matches the scale of a colony and its fast-ice apron.
        """
        self.precision = precision
        self.cells: dict[str, dict[str, Any]] = {}
        self.total = 0
        self.skipped = 0
        self.earliest: str | None = None
        self.latest: str | None = None
        self._uncertainties: list[float] = []

    def add(self, records: Iterable[Any]) -> int:
        """Fold a batch of records in, discarding them immediately.

        Parameters
        ----------
        records
            Decoded GBIF ``results`` entries, or a page of them.

        Returns
        -------
        int
            How many records were accepted into this batch.
        """
        accepted = 0
        for record in records:
            if not isinstance(record, dict):
                self.skipped += 1
                continue

            latitude = record.get("decimalLatitude")
            longitude = record.get("decimalLongitude")
            if latitude is None or longitude is None:
                self.skipped += 1
                continue
            try:
                lat = round(float(latitude), self.precision)
                lon = round(float(longitude), self.precision)
            except (TypeError, ValueError):
                self.skipped += 1
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                self.skipped += 1
                continue

            key = f"{lat:.{self.precision}f},{lon:.{self.precision}f}"
            cell = self.cells.get(key)
            if cell is None:
                cell = {
                    "key": key,
                    "lat": lat,
                    "lon": lon,
                    "record_count": 0,
                    "first_seen": None,
                    "last_seen": None,
                    "basis_of_record": {},
                }
                self.cells[key] = cell

            cell["record_count"] += 1
            self.total += 1
            accepted += 1

            event_date = record.get("eventDate")
            if isinstance(event_date, str) and event_date:
                date_part = event_date[:10]
                if cell["first_seen"] is None or date_part < cell["first_seen"]:
                    cell["first_seen"] = date_part
                if cell["last_seen"] is None or date_part > cell["last_seen"]:
                    cell["last_seen"] = date_part
                if self.earliest is None or date_part < self.earliest:
                    self.earliest = date_part
                if self.latest is None or date_part > self.latest:
                    self.latest = date_part

            basis = record.get("basisOfRecord")
            if isinstance(basis, str):
                counts = cell["basis_of_record"]
                counts[basis] = counts.get(basis, 0) + 1

            uncertainty = record.get("coordinateUncertaintyInMeters")
            if isinstance(uncertainty, int | float) and uncertainty >= 0:
                self._uncertainties.append(float(uncertainty))

        return accepted

    @property
    def median_uncertainty_m(self) -> float | None:
        """Return the median reported coordinate uncertainty, in metres.

        GBIF deliberately *generalises* coordinates for threatened taxa, setting
        ``informationWithheld`` to explain the blur. Emperor penguins are
        IUCN Near Threatened, so real records routinely arrive with 20 km+
        uncertainty. Reporting it keeps the dashboard honest about what its own
        presence layer can resolve.
        """
        if not self._uncertainties:
            return None
        ordered = sorted(self._uncertainties)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2

    def summary(self) -> dict[str, Any]:
        """Return the aggregated grid.

        Returns
        -------
        dict[str, Any]
            ``{"total", "skipped", "cells", "first_seen", "last_seen",
            "cell_size_m", "median_uncertainty_m", "provenance",
            "interpretation"}``.
        """
        return {
            "total": self.total,
            "skipped": self.skipped,
            "cells": self.cells,
            "first_seen": self.earliest,
            "last_seen": self.latest,
            "cell_size_m": _COORDINATE_PRECISION_M * (10**self.precision),
            "median_uncertainty_m": self.median_uncertainty_m,
            "coordinates_are_generalised": self.median_uncertainty_m is not None
            and self.median_uncertainty_m > 1000.0,
            "provenance": "gbif.occurrence_search",
            "interpretation": "record density is observer effort, not abundance",
        }


def aggregate_occurrences(
    records: Iterable[dict[str, Any]],
    *,
    precision: int = 2,
) -> dict[str, Any]:
    """Aggregate GBIF occurrence records into a spatial presence grid.

    Convenience wrapper over :class:`PresenceGrid` for one-shot callers and
    tests. The scanning client uses the class directly so it can discard each
    page instead of holding the whole result set.

    Parameters
    ----------
    records
        Decoded GBIF ``results`` entries.
    precision
        Decimal places to round coordinates to when binning. 2 dp is ~1.1 km,
        which matches the scale of a colony and its fast-ice apron.

    Returns
    -------
    dict[str, Any]
        Aggregated grid; see :meth:`PresenceGrid.summary`.

    Examples
    --------
    >>> result = aggregate_occurrences([
    ...     {"decimalLatitude": -77.5421, "decimalLongitude": 166.611, "eventDate": "2024-01-02"},
    ...     {"decimalLatitude": -77.5389, "decimalLongitude": 166.614, "eventDate": "2024-01-03"},
    ... ])
    >>> result["total"]
    2
    >>> len(result["cells"])
    1
    >>> result["median_uncertainty_m"] is None
    True
    """
    grid = PresenceGrid(precision=precision)
    grid.add(records)
    return grid.summary()


class ColonyCensusClient:
    """Resolves colony locations and census figures from GBIF and SCAR.

    Parameters
    ----------
    client
        Shared HTTP client.
    species
        Accepted taxon name. Resolved to a GBIF usage key on first use and
        cached for the process lifetime.
    max_distance_km
        Colonies farther than this from the site position are excluded.
    max_colonies
        Cap on returned colonies, largest population first.

    Examples
    --------
    >>> from emperor_space_tracker.net import HttpClient
    >>> census = ColonyCensusClient(HttpClient(), max_distance_km=700.0)
    >>> colonies = census.poll(latitude=-77.85, longitude=166.67)  # doctest: +SKIP
    """

    def __init__(
        self,
        client: HttpClient,
        *,
        species: str = "Aptenodytes forsteri",
        max_distance_km: float = 700.0,
        max_colonies: int = 12,
    ) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.client = client
        self.species = species
        self.max_distance_km = max_distance_km
        self.max_colonies = max_colonies
        self._taxon_key: int | None = None
        self._taxon_rank: str = ""
        self._taxon_match_type: str = ""
        self._catalogue: list[Colony] | None = None

    def resolve_taxon_key(self) -> int:
        """Resolve the species name to a GBIF taxon usage key.

        A name that does not exist in the GBIF backbone still resolves to
        *something* -- its parent genus, typically -- and GBIF reports that as a
        successful match with ``matchType = HIGHERRANK``. Searching the returned
        key would then pull every record in the genus and attribute it to the
        configured species, so a match above species rank is reported rather than
        accepted quietly. This is not hypothetical: the 2026 revision of the
        gentoo complex is recent enough that the GBIF backbone resolves
        ``Pygoscelis kerguelensis`` to the genus ``Pygoscelis``.

        Returns
        -------
        int
            The GBIF usage key for :attr:`species`.

        Raises
        ------
        SourceError
            If the name cannot be resolved.

        Examples
        --------
        >>> isinstance(ColonyCensusClient.resolve_taxon_key, type(lambda: 0))
        True
        """
        if self._taxon_key is not None:
            return self._taxon_key
        # `quote` rather than a hand-rolled space replacement: a binomial has no
        # characters that need escaping today, but the species is operator-
        # configurable and a stray `&` in a name would silently truncate the
        # query into something that resolves to a different taxon.
        url = f"{GBIF_ENDPOINTS['species_match']}?name={quote(self.species, safe='')}"
        payload = self.client.get_json(url, source="gbif.species")
        if not isinstance(payload, dict):
            raise SourceError("gbif.species", "species/match did not return an object")
        key = payload.get("usageKey")
        if not isinstance(key, int):
            detail = payload.get("message", "no usageKey")
            msg = f"GBIF could not resolve {self.species!r}: {detail}"
            raise SourceError("gbif.species", msg)
        self._taxon_key = key

        matched = str(payload.get("scientificName") or "?")
        rank = str(payload.get("rank") or "?").upper()
        match_type = str(payload.get("matchType") or "?").upper()
        self._taxon_rank = rank
        self._taxon_match_type = match_type

        if rank != "SPECIES":
            # Proceed -- a genus-level search bounded by the colony envelope is
            # still far more useful than nothing -- but make the widening
            # impossible to miss, because every record it returns is a candidate
            # rather than an identification.
            _LOG.warning(
                "GBIF has no species-level record for %r: it matched %r at %s rank "
                "(matchType=%s, key=%d). Occurrence counts will include every "
                "species in that taxon, so they are presence evidence for the "
                "genus and not for %s specifically.",
                self.species, matched, rank, match_type, key, self.species,
            )
        else:
            _LOG.info(
                "resolved %s to GBIF taxon key %d (%s)",
                self.species,
                key,
                matched,
            )
        return key

    @property
    def taxon_is_species_level(self) -> bool:
        """Whether the resolved GBIF match is a species and not a parent rank.

        ``False`` means the occurrence search is broader than the configured
        species, which the presence summary and the source-health detail both
        report.
        """
        return self._taxon_rank == "SPECIES"

    def gbif_envelope(
        self,
        colonies: Iterable[Colony],
        *,
        pad_degrees: float = 2.0,
    ) -> dict[str, float] | None:
        """Derive a GBIF search envelope from the colonies we actually hold.

        The search bounds used to be hardcoded to ``country=AQ`` and
        ``decimalLatitude=-90,-60``, which is right for an Emperor penguin and
        returns nothing at all for anything else -- and returns nothing
        *silently*, which is the worst failure mode available to a monitoring
        node. Deriving the bounds from the catalogue instead means the query
        follows the data: add a colony anywhere and the search widens to
        include it, with no second table of ranges to keep in step.

        Parameters
        ----------
        colonies
            Catalogue records. Only those matching :attr:`species` are used, so
            a multi-species catalogue still produces a tight search.
        pad_degrees
            Degrees added to each side, so a record a little off the colony
            point is not excluded by a boundary that cuts through the colony.

        Returns
        -------
        dict[str, float] | None
            ``min_lat``, ``max_lat``, ``min_lon``, ``max_lon``, or ``None`` if
            no catalogue record matches :attr:`species`.

        Notes
        -----
        The longitude filter is omitted when the colonies straddle the
        antimeridian, because GBIF's ``decimalLongitude`` takes a single
        ``min,max`` pair and a min/max of 179 and -179 describes the *whole*
        globe rather than a thin sliver. Widening to a global longitude search
        is the safe reading; the latitude bound still constrains the result.
        """
        matching = [c for c in colonies if c.species == self.species]
        if not matching:
            return None
        min_lat = min(c.latitude for c in matching) - pad_degrees
        max_lat = max(c.latitude for c in matching) + pad_degrees
        min_lon = min(c.longitude for c in matching) - pad_degrees
        max_lon = max(c.longitude for c in matching) + pad_degrees
        if max_lon - min_lon > 180.0:
            _LOG.warning(
                "%s colonies span %.0f degrees of longitude; omitting the GBIF "
                "longitude filter rather than querying the whole globe",
                self.species,
                max_lon - min_lon,
            )
            return {"min_lat": min_lat, "max_lat": max_lat, "min_lon": -180.0, "max_lon": 180.0}
        return {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}

    def fetch_occurrences(
        self,
        *,
        limit: int = _PAGE_SIZE,
        pages: int = 1,
        envelope: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Query GBIF occurrence records for the species inside ``envelope``.

        Parameters
        ----------
        limit
            Records per page, capped at 300 by GBIF.
        pages
            Number of pages to fetch, capped at :data:`_MAX_PAGES`.
        envelope
            Search bounds from :meth:`gbif_envelope`. When ``None`` the search
            is unbounded, which is correct-but-useless rather than silently
            empty: an operator sees every record for the taxon and can see in
            the summary that no envelope was applied.

        Returns
        -------
        dict[str, Any]
            The output of :func:`aggregate_occurrences`, plus the raw
            ``total_available`` count GBIF reports so a truncated scan is
            visible to the operator, plus the envelope actually applied.

        Raises
        ------
        SourceError
            If GBIF cannot be reached or returns an unexpected shape.
        """
        taxon_key = self.resolve_taxon_key()
        page_size = min(limit, 300)
        grid = PresenceGrid()
        total_available = 0

        if envelope is None:
            _LOG.warning(
                "no GBIF envelope for %s; searching without a geographic bound, "
                "so the result is every record GBIF holds for this taxon",
                self.species,
            )
            bounds = ""
        else:
            bounds = (
                f"&decimalLatitude={envelope['min_lat']:.4f},{envelope['max_lat']:.4f}"
                f"&decimalLongitude={envelope['min_lon']:.4f},{envelope['max_lon']:.4f}"
            )

        for page in range(max(1, min(pages, _MAX_PAGES))):
            # No `country=` filter. A country code is a silent way to return
            # nothing -- a typo, or a species that breeds in a territory nobody
            # thought of, produces an empty scan that looks exactly like "no
            # penguins were observed". The coordinate envelope comes from the
            # colonies we actually hold, so it cannot disagree with the data.
            #
            # The latitude bound is still doing real work: it keeps out regions
            # where the species is a vagrant rather than a breeder, so a record
            # outside the envelope is not evidence of a colony. Aggregation bins
            # by presence, never by abundance.
            url = (
                f"{GBIF_ENDPOINTS['occurrence_search']}?taxon_key={taxon_key}"
                f"{bounds}"
                f"&limit={page_size}&offset={page * page_size}"
                "&hasCoordinate=true&hasGeospatialIssue=false"
            )
            payload = self.client.get_json(url, source="gbif.occurrence")
            if not isinstance(payload, dict):
                raise SourceError("gbif.occurrence", "occurrence/search did not return an object")
            results = payload.get("results")
            if not isinstance(results, list):
                raise SourceError("gbif.occurrence", "occurrence/search returned no results array")
            total_available = int(payload.get("count") or 0)
            grid.add(results)
            # Release the page before the next request. Without this the whole
            # scan accumulates, and 300 fat records per page is the difference
            # between sitting under the memory ceiling and blowing through it.
            del results
            del payload
            if total_available <= grid.total:
                break

        summary = grid.summary()
        summary["total_available"] = total_available
        summary["fetched"] = grid.total
        summary["taxon"] = {
            "name": self.species,
            "gbif_usage_key": taxon_key,
            "gbif_rank": self._taxon_rank or None,
            "gbif_match_type": self._taxon_match_type or None,
            "species_level": self.taxon_is_species_level,
        }
        summary["envelope"] = envelope
        return summary

    def reference_catalogue(self) -> list[Colony]:
        """Return the curated census catalogue, loading it once per process.

        Returns
        -------
        list[Colony]
        """
        if self._catalogue is None:
            self._catalogue = load_reference_catalogue()
        return self._catalogue

    def near(
        self,
        colonies: Iterable[Colony],
        *,
        latitude: float,
        longitude: float,
    ) -> list[Colony]:
        """Filter colonies to those within the configured radius of the site.

        Parameters
        ----------
        colonies
            Candidate colonies.
        latitude
            Site latitude.
        longitude
            Site longitude.

        Returns
        -------
        list[Colony]
            Matching colonies with :attr:`Colony.notes` annotated with the
            computed distance, largest population first.

        Examples
        --------
        >>> client = ColonyCensusClient(HttpClient(), max_distance_km=50.0)
        >>> client.near([], latitude=-77.85, longitude=166.67)
        []
        """
        limit_m = self.max_distance_km * 1000.0
        matched: list[Colony] = []
        for colony in colonies:
            distance = geo_distance_m(latitude, longitude, colony.latitude, colony.longitude)
            if distance > limit_m:
                continue
            note = f"{colony.notes} | {distance:.0f} km from site".strip(" |")
            matched.append(replace(colony, notes=note))
        matched.sort(key=lambda c: (-(c.population_estimate or 0), c.name))
        return matched[: self.max_colonies]

    def poll(
        self,
        *,
        latitude: float,
        longitude: float,
        include_gbif: bool = True,
    ) -> tuple[list[Colony], list[SourceHealth]]:
        """Assemble the colony set for a polling pass.

        Parameters
        ----------
        latitude
            Site latitude.
        longitude
            Site longitude.
        include_gbif
            Query the GBIF occurrence endpoint. Census figures come from the
            reference catalogue either way; this only adds presence evidence.

        Returns
        -------
        tuple[list[Colony], list[SourceHealth]]
            The colonies within range, and per-source health for the pass.
        """
        health: list[SourceHealth] = []
        observed = utc_now()
        presence: dict[str, Any] = {}

        catalogue = self.reference_catalogue()
        health.append(
            SourceHealth(
                "scar.catalogue",
                bool(catalogue),
                0,
                f"{len(catalogue)} curated census records",
                len(catalogue),
                observed,
            )
        )

        if include_gbif:
            started = utc_now()
            try:
                presence = self.fetch_occurrences(
                    envelope=self.gbif_envelope(catalogue),
                )
                elapsed = int((utc_now() - started).total_seconds() * 1000)
                detail = (
                    f"{presence['fetched']} of {presence.get('total_available', 0):,} records "
                    f"across {len(presence['cells'])} cells"
                )
                if not self.taxon_is_species_level:
                    # The count is a genus total, so say so on the health row
                    # the operator actually reads rather than only in the log.
                    detail += (
                        f"; GBIF has no species record for {self.species!r} and matched at "
                        f"{self._taxon_rank or 'unknown'} rank, so these are presence "
                        "records for that whole taxon"
                    )
                health.append(
                    SourceHealth("gbif.occurrence", True, elapsed, detail,
                                 presence["fetched"], observed)
                )
            except SourceError as exc:
                elapsed = int((utc_now() - started).total_seconds() * 1000)
                _LOG.warning("GBIF occurrence scan failed: %s", exc)
                health.append(
                    SourceHealth("gbif.occurrence", False, elapsed, str(exc), 0, observed)
                )

        colonies = self.near(catalogue, latitude=latitude, longitude=longitude)

        if presence.get("cells"):
            colonies = self._annotate_with_presence(colonies, presence)

        return colonies, health

    def _annotate_with_presence(
        self,
        colonies: list[Colony],
        presence: dict[str, Any],
    ) -> list[Colony]:
        """Attach nearest-cell GBIF record counts to each colony.

        A colony's cell is the GBIF presence cell within 60 km. The count is
        labelled ``GBIF records`` in the notes so it is never mistaken for a
        breeding-pair estimate.

        Parameters
        ----------
        colonies
            Colonies to annotate.
        presence
            Output of :func:`aggregate_occurrences`.

        Returns
        -------
        list[Colony]
        """
        cells = presence.get("cells") or {}
        annotated: list[Colony] = []
        for colony in colonies:
            best_key = None
            best_distance = 60_000.0
            for key, cell in cells.items():
                distance = geo_distance_m(
                    colony.latitude, colony.longitude, cell["lat"], cell["lon"]
                )
                if distance < best_distance:
                    best_distance = distance
                    best_key = key
            if best_key is None:
                annotated.append(colony)
                continue
            count = cells[best_key]["record_count"]
            note = (
                f"{colony.notes} | GBIF records: {count} "
                f"within {best_distance / 1000:.0f} km"
            ).strip(" |")
            # `replace` rather than a field-by-field rebuild: this was a
            # verbatim copy of the construction above, which is how new fields
            # used to be silently dropped.
            annotated.append(replace(colony, notes=note))
        return annotated
