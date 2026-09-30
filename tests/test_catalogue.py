"""Validation for the bundled colony catalogue and the species model.

Every record in ``data/colonies.toml`` ships to every node, and until this file
existed nothing in the test suite ever parsed it. A typo in a census year or a
latitude would have reached a deployed field node untouched, and the file's
whole purpose is to be trusted precisely because someone reviewed it.

The tests are deliberately about *provenance and self-consistency* rather than
about whether a number is right -- nobody can check that from a test. What can
be checked is that a record is complete, internally consistent, and honest
about what it does not know.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from emperor_space_tracker.config import Config, SarConfig
from emperor_space_tracker.models import BreedingHabitat, Colony
from emperor_space_tracker.net import HttpClient
from emperor_space_tracker.sources.biological import (
    ColonyCensusClient,
    load_reference_catalogue,
)
from emperor_space_tracker.sources.sar import SarClient

CATALOGUE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "emperor_space_tracker"
    / "data"
    / "colonies.toml"
)


@pytest.fixture(scope="module")
def catalogue() -> list[Colony]:
    """Parse the shipped catalogue once for the whole module."""
    return load_reference_catalogue()


# --------------------------------------------------------------------------- #
# Every record is complete and self-consistent
# --------------------------------------------------------------------------- #


def test_every_shipped_record_parses() -> None:
    """The catalogue loads at all, and is not silently empty.

    An empty result is what a missing file produces, and a missing data file in
    an installed wheel degrades to GBIF-only mapping with a warning rather than
    an error -- so a packaging regression would look like a working install.
    """
    colonies = load_reference_catalogue()
    assert len(colonies) >= 10, f"expected a real catalogue, got {len(colonies)} records"


def test_identifiers_are_unique_and_prefixed_by_species(catalogue: list[Colony]) -> None:
    """Ids are unique, and their prefix agrees with the record's taxon.

    The prefix is a cheap guard against exactly the error this change exists to
    prevent: a gentoo record filed under an ``cpe-`` (Emperor) id, where nothing
    downstream would notice.
    """
    ids = [c.colony_id for c in catalogue]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    assert not duplicates, f"duplicate colony ids: {duplicates}"

    prefixes = {
        "Aptenodytes forsteri": "cpe-",
        "Pygoscelis kerguelensis": "pgk-",
    }
    for colony in catalogue:
        expected = prefixes.get(colony.species)
        if expected is None:
            continue
        assert colony.colony_id.startswith(expected), (
            f"{colony.colony_id} is a {colony.species} record but its id does not "
            f"start with {expected!r}"
        )


@pytest.mark.parametrize(
    ("field", "low", "high"),
    [
        ("latitude", -90.0, 90.0),
        ("longitude", -180.0, 180.0),
    ],
)
def test_coordinates_are_in_range(
    catalogue: list[Colony], field: str, low: float, high: float
) -> None:
    """Every coordinate is a real one.

    A latitude that slipped a sign would put a colony on the wrong side of the
    planet and, for a fast-ice breeder, past the -60 degree gate that decides
    whether SAR will image it at all.
    """
    for colony in catalogue:
        value = getattr(colony, field)
        assert low <= value <= high, f"{colony.colony_id}.{field} = {value} is out of range"


def test_fast_ice_ratio_is_only_present_for_fast_ice_breeders(
    catalogue: list[Colony],
) -> None:
    """The ice-access figure is present exactly for the species it describes.

    ``fast_ice_ratio`` is only meaningful for a species that nests on sea ice. A
    land-nesting record carrying one would be asserting a thing nobody measured,
    and the dashboard would render it as a real percentage.
    """
    for colony in catalogue:
        if colony.breeding_habitat is BreedingHabitat.FAST_ICE:
            assert colony.fast_ice_ratio is not None, (
                f"{colony.colony_id} breeds on fast ice but has no fast_ice_ratio"
            )
        else:
            assert colony.fast_ice_ratio is None, (
                f"{colony.colony_id} nests on land but carries fast_ice_ratio="
                f"{colony.fast_ice_ratio}, which is not a property of its habitat"
            )


def test_fast_ice_ratios_are_fractions_not_percentages(catalogue: list[Colony]) -> None:
    """Ice-access values lie in [0, 1]."""
    for colony in catalogue:
        if colony.fast_ice_ratio is not None:
            assert 0.0 <= colony.fast_ice_ratio <= 1.0, (
                f"{colony.colony_id}.fast_ice_ratio = {colony.fast_ice_ratio}"
            )


def test_populations_are_attached_to_a_year_and_a_source(catalogue: list[Colony]) -> None:
    """No count appears without the date and provenance that make it usable.

    A breeding-pair figure with no year is the specific misuse this file exists
    to prevent, and the store has no way to recover the year once it is lost.
    """
    for colony in catalogue:
        if colony.population_estimate is None:
            continue
        assert colony.population_year is not None, (
            f"{colony.colony_id} has a population but no population_year"
        )
        assert colony.population_source.strip(), (
            f"{colony.colony_id} has a population but no population_source"
        )
        assert colony.population_estimate >= 0, colony.colony_id


def test_a_population_year_is_plausible(catalogue: list[Colony]) -> None:
    """Census years are inside the plausible band for a colony record.

    A transposed year such as 1922 or 2202 is a typo, not a data point, and
    nothing else in the pipeline would catch it.
    """
    for colony in catalogue:
        if colony.population_year is None:
            continue
        assert 1900 <= colony.population_year <= 2026, (
            f"{colony.colony_id}.population_year = {colony.population_year} is implausible"
        )


def test_every_record_names_a_species_and_a_habitat(catalogue: list[Colony]) -> None:
    """No record is anonymous.

    The loader requires both, so this also pins that the requirement is
    enforced rather than defaulted: a record that omitted ``species`` would have
    raised rather than silently becoming an Emperor.
    """
    for colony in catalogue:
        assert len(colony.species.split()) == 2, (
            f"{colony.colony_id}.species = {colony.species!r} is not a binomial"
        )
        assert isinstance(colony.breeding_habitat, BreedingHabitat), colony.colony_id


def test_shipped_catalogue_covers_both_habitats(catalogue: list[Colony]) -> None:
    """The catalogue exercises the fast-ice and land paths.

    If it only held one species, the SAR skip and the rule filter would never
    run against real data, and the whole point of the habitat field would be
    untested in production. This also guards against a refactor quietly dropping
    the gentoo records.
    """
    habitats = {c.breeding_habitat for c in catalogue}
    assert habitats == {BreedingHabitat.FAST_ICE, BreedingHabitat.LAND}, habitats
    assert len({c.species for c in catalogue}) >= 2


# --------------------------------------------------------------------------- #
# The record must say what it does not know
# --------------------------------------------------------------------------- #


def test_uncertain_assignments_record_their_uncertainty(catalogue: list[Colony]) -> None:
    """An inferred species placement is labelled as inferred in the record.

    Heard Island is assigned to the southeastern gentoo on tracking evidence
    rather than on genotypes, because the 2026 revision has no molecular data
    from the island. A record that asserted the placement without saying so
    would be a stronger claim than the evidence supports.
    """
    heard = [c for c in catalogue if c.colony_id == "pgk-heard-island"]
    assert heard, "the Heard Island record should be present"
    notes = heard[0].notes.lower()
    assert "inferential" in notes or "inferred" in notes, (
        "an assign-by-inference record must say so in its notes"
    )


def test_stale_censuses_are_dated_not_hidden(catalogue: list[Colony]) -> None:
    """Every record's census year survives round-tripping through the model.

    A count whose year is lost in translation becomes a current claim about a
    population that was last counted decades ago.
    """
    for colony in catalogue:
        if colony.population_estimate is not None:
            assert colony.population_year is not None, colony.colony_id
        if colony.last_census_at is not None:
            assert colony.last_census_at.tzinfo is not None, (
                f"{colony.colony_id}.last_census_at must be timezone-aware"
            )


# --------------------------------------------------------------------------- #
# Downstream gates must actually see both kinds
# --------------------------------------------------------------------------- #


def test_sar_skips_land_nesting_colonies(catalogue: list[Colony]) -> None:
    """A catalogue of only gentoo colonies produces no SAR scenes.

    This is the load-bearing behaviour for a non-fast-ice species. Without it
    the node would place a confident sigma0 reading over a beach, and the
    classification bands would report open water over a colony that is not on
    any ice at all.
    """
    land = [c for c in catalogue if not c.monitors_fast_ice]
    assert land, "need at least one land-nesting colony for this test"
    client = SarClient(HttpClient(), SarConfig(backend="synthetic", grid_cells=9))
    scenes, health = client.poll(colonies=land)
    assert scenes == []
    assert len(health) == 1
    assert health[0].ok
    assert "does not apply" in health[0].detail


def test_sar_still_images_fast_ice_colonies(catalogue: list[Colony]) -> None:
    """The skip is selective, not a blanket disable."""
    ice = [c for c in catalogue if c.monitors_fast_ice]
    assert ice
    client = SarClient(HttpClient(), SarConfig(backend="synthetic", grid_cells=9))
    scenes, _ = client.poll(colonies=ice, scenes_per_colony=1)
    assert scenes, "fast-ice breeders must still be imaged"
    assert {s.colony_id for s in scenes} == {c.colony_id for c in ice}


def test_sar_mixed_catalogue_images_only_the_ice_colonies(catalogue: list[Colony]) -> None:
    """A mixed colony list images the ice breeders and reports the rest skipped."""
    client = SarClient(HttpClient(), SarConfig(backend="synthetic", grid_cells=9))
    scenes, health = client.poll(colonies=catalogue, scenes_per_colony=1)
    imaged = {s.colony_id for s in scenes}
    assert imaged == {c.colony_id for c in catalogue if c.monitors_fast_ice}
    assert any("skipped" in h.detail for h in health), health


def test_gbif_envelope_follows_the_configured_species(catalogue: list[Colony]) -> None:
    """The GBIF search box is derived from the colonies held, not hardcoded.

    The bounds used to be ``country=AQ`` and ``-90,-60``, which is correct for an
    Emperor and returns nothing at all -- silently -- for anything else.
    """
    client = ColonyCensusClient(HttpClient(), species="Pygoscelis kerguelensis")
    envelope = client.gbif_envelope(catalogue)
    assert envelope is not None
    # The Kerguelen and Heard records sit near 49-53 S, 70-73 E. An Emperor
    # envelope would be entirely south of -60 and would exclude every one.
    assert envelope["max_lat"] > -60, envelope
    assert 60 < envelope["min_lon"] < 80, envelope


def test_gbif_envelope_excludes_other_species(catalogue: list[Colony]) -> None:
    """Asking about one species does not widen the box to cover the other."""
    client = ColonyCensusClient(HttpClient(), species="Aptenodytes forsteri")
    envelope = client.gbif_envelope(catalogue)
    assert envelope is not None
    assert envelope["max_lat"] < -60, envelope
    assert envelope["max_lon"] > 160, envelope


def test_gbif_envelope_is_none_for_an_unknown_species(catalogue: list[Colony]) -> None:
    """An unheld species yields no envelope rather than a wrong one.

    Returning ``None`` makes ``fetch_occurrences`` search unbounded and say so
    in a warning, which is recoverable and visible. Inventing a box would be
    neither.
    """
    client = ColonyCensusClient(HttpClient(), species="Pygoscelis adeliae")
    assert client.gbif_envelope(catalogue) is None


def test_gbif_envelope_handles_the_antimeridian() -> None:
    """Colonies spanning 180 degrees do not produce a whole-globe filter.

    ``min``/``max`` of 179 and -179 describes every longitude, not a thin
    sliver, so the longitude bound is dropped and only latitude constrains.
    """
    def make(lat: float, lon: float) -> Colony:
        return Colony(
            colony_id=f"c-{lat}-{lon}", name=f"c{lat}{lon}", latitude=lat, longitude=lon,
            region="test", population_estimate=None, population_year=None,
            population_source="test", species="Testus testus",
            breeding_habitat=BreedingHabitat.FAST_ICE,
        )

    client = ColonyCensusClient(HttpClient(), species="Testus testus")
    envelope = client.gbif_envelope(
        [make(-78.0, 179.0), make(-66.0, -179.0)], pad_degrees=1.0
    )
    assert envelope is not None
    assert envelope["min_lon"] == -180.0
    assert envelope["max_lon"] == 180.0
    # The latitude bound is still the real filter, so the search is not
    # unbounded even though the longitude one was dropped.
    assert envelope["min_lat"] == pytest.approx(-79.0)
    assert envelope["max_lat"] == pytest.approx(-65.0)


def test_gbif_envelope_keeps_a_tight_longitude_bound_within_a_hemisphere(
    catalogue: list[Colony],
) -> None:
    """The ordinary case keeps both bounds, so the search stays small."""
    client = ColonyCensusClient(HttpClient(), species="Aptenodytes forsteri")
    envelope = client.gbif_envelope(catalogue)
    assert envelope is not None
    # The Emperor colonies run from Cape Adare (~166 E) round to the Mawson
    # Coast (~63 E), so the box has to cross the prime meridian and still stay
    # well short of a whole globe.
    assert envelope["min_lon"] < envelope["max_lon"]
    assert (envelope["max_lon"] - envelope["min_lon"]) < 180


def test_a_genus_level_gbif_match_is_reported_not_silently_accepted() -> None:
    """A parent-rank taxon match is surfaced, not treated as a species match.

    The GBIF backbone has no species record for *Pygoscelis kerguelensis* -- the
    2026 revision is too recent -- so `species/match` answers with the genus
    `Pygoscelis` and `matchType=HIGHERRANK`. Searching that key returns every
    *Pygoscelis* record in the envelope, so presenting the count as species
    presence would be a silent misattribution. The client proceeds (a bounded
    genus scan beats nothing) but records that it is not species-level.
    """
    class _Stub:
        """A client that returns a GBIF genus-level match."""

        def get_json(self, url: str, *, source: str) -> dict[str, object]:
            """Answer with the genus, as the live backbone does."""
            assert "Pygoscelis%20kerguelensis" in url
            return {
                "usageKey": 2481662,
                "scientificName": "Pygoscelis Wagler, 1832",
                "rank": "GENUS",
                "matchType": "HIGHERRANK",
            }

    client = ColonyCensusClient(_Stub(), species="Pygoscelis kerguelensis")  # type: ignore[arg-type]
    assert client.resolve_taxon_key() == 2481662
    assert client.taxon_is_species_level is False


def test_an_exact_species_match_is_reported_as_species_level() -> None:
    """The ordinary case is still recognised as a species match."""

    class _Stub:
        """A client that returns an exact species match."""

        def get_json(self, url: str, *, source: str) -> dict[str, object]:
            """Answer with the species, as the live backbone does."""
            return {
                "usageKey": 2481661,
                "scientificName": "Aptenodytes forsteri G.R.Gray, 1844",
                "rank": "SPECIES",
                "matchType": "EXACT",
            }

    client = ColonyCensusClient(_Stub(), species="Aptenodytes forsteri")  # type: ignore[arg-type]
    assert client.resolve_taxon_key() == 2481661
    assert client.taxon_is_species_level is True


# --------------------------------------------------------------------------- #
# Negative cases for the loader
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "colonies.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_record_without_a_species_is_rejected(tmp_path: Path) -> None:
    """Omitting the taxon is an error, not a default.

    Defaulting to the Emperor is how a gentoo record ends up evaluated against
    fast-ice thresholds that mean nothing for it.
    """
    from emperor_space_tracker.errors import SourceError

    path = _write(
        tmp_path,
        """
[[colony]]
id = "x-1"
name = "Nowhere"
latitude = -50.0
longitude = 70.0
region = "test"
population = 100
population_year = 2018
population_source = "test"
breeding_habitat = "land"
""",
    )
    with pytest.raises(SourceError, match="malformed record"):
        load_reference_catalogue(path)


def test_an_unknown_breeding_habitat_is_rejected(tmp_path: Path) -> None:
    """A habitat the model does not have is a hard error.

    Silently coercing it to a default would put a colony into the wrong
    monitoring path, which is the specific mistake worth being loud about.
    """
    from emperor_space_tracker.errors import SourceError

    path = _write(
        tmp_path,
        """
[[colony]]
id = "x-1"
name = "Nowhere"
latitude = -50.0
longitude = 70.0
region = "test"
population = 100
population_year = 2018
population_source = "test"
species = "Testus testus"
breeding_habitat = "pack_ice"
""",
    )
    with pytest.raises(SourceError, match="malformed record"):
        load_reference_catalogue(path)


def test_a_missing_catalogue_degrades_to_empty(tmp_path: Path) -> None:
    """A missing file is empty, not an error.

    A stripped install must still start and fall back to GBIF presence mapping.
    """
    assert load_reference_catalogue(tmp_path / "nope.toml") == []


# --------------------------------------------------------------------------- #
# The fail-closed rule: a wide bind without auth is a hard error
# --------------------------------------------------------------------------- #


def test_a_wide_bind_without_auth_is_rejected() -> None:
    """The dashboard must not be reachable with authentication switched off.

    Nothing about the rendered page tells an operator their store is readable by
    anyone who can reach the port, so this is refused at config load rather than
    warned about. It is the state this project has spent its whole life trying
    to avoid.
    """
    from emperor_space_tracker.config import DashboardAuthConfig, DashboardConfig
    from emperor_space_tracker.errors import ConfigError

    # A wildcard, the IPv6 wildcard, and a private mesh address: all three
    # make the dashboard reachable off the machine.
    for host in ("0.0.0.0", "::", "100.112.18.30"):  # noqa: S104
        config = Config(
            dashboard=DashboardConfig(
                host=host, auth=DashboardAuthConfig(enabled=False)
            )
        )
        with pytest.raises(ConfigError, match="reachable off this machine"):
            config.validate()


def test_a_wide_bind_with_auth_is_accepted() -> None:
    """The same bind is fine once authentication is on."""
    from emperor_space_tracker.config import DashboardAuthConfig, DashboardConfig

    config = Config(
        dashboard=DashboardConfig(
            host="0.0.0.0", auth=DashboardAuthConfig(enabled=True)  # noqa: S104
        )
    )
    assert config.validate().dashboard.auth.enabled is True


def test_loopback_without_auth_is_fine() -> None:
    """Loopback is the coherent unauthenticated configuration.

    Nothing but this machine can open the socket, so authentication would be
    defending against nothing.
    """
    from emperor_space_tracker.config import DashboardAuthConfig, DashboardConfig

    config = Config(
        dashboard=DashboardConfig(
            host="127.0.0.1", auth=DashboardAuthConfig(enabled=False)
        )
    )
    assert config.validate().dashboard.bind_is_not_loopback is False


def test_a_disabled_dashboard_skips_the_auth_requirement() -> None:
    """A node that runs no dashboard at all is not asked to configure auth."""
    from emperor_space_tracker.config import DashboardAuthConfig, DashboardConfig

    config = Config(
        dashboard=DashboardConfig(
            enabled=False,
            host="0.0.0.0",  # noqa: S104
            auth=DashboardAuthConfig(enabled=False),
        )
    )
    assert config.validate().dashboard.enabled is False


def test_the_lockout_cap_must_not_be_below_its_base() -> None:
    """A cap under the base would mean the first lockout is silently clipped."""
    from emperor_space_tracker.config import DashboardAuthConfig, DashboardConfig
    from emperor_space_tracker.errors import ConfigError

    config = Config(
        dashboard=DashboardConfig(
            auth=DashboardAuthConfig(lockout_base_seconds=600.0, lockout_cap_seconds=60.0)
        )
    )
    with pytest.raises(ConfigError, match="lockout_cap_seconds"):
        config.validate()


def test_a_nested_auth_section_round_trips_through_the_config_file(
    tmp_path: Path,
) -> None:
    """`[dashboard.auth]` must reach the dataclass, not sit in a dict.

    TOML nests the table, so the loader has to walk into it. A dotted section
    name in the loader's map is what makes that work, and a mistake here would
    leave `config.dashboard.auth` as a raw dict -- so that it fails loudly rather
    than at the first `.enabled` access deep in the dashboard.
    """
    from emperor_space_tracker.config import DashboardAuthConfig, load_config

    path = tmp_path / "config.toml"
    path.write_text(
        "[dashboard]\n"
        'host = "100.112.18.30"\n'
        "\n[dashboard.auth]\n"
        "enabled = true\n"
        "max_failures = 9\n"
        "lockout_base_seconds = 30.0\n",
        encoding="utf-8",
    )
    config = load_config(path, use_user_config=False)
    assert isinstance(config.dashboard.auth, DashboardAuthConfig)
    assert config.dashboard.auth.enabled is True
    assert config.dashboard.auth.max_failures == 9
    assert config.dashboard.auth.lockout_base_seconds == 30.0
    # Inherited from the packaged defaults, not lost with the override.
    assert config.dashboard.auth.lockout_cap_seconds == 900.0
    assert config.dashboard.auth.proxy_secret_env == "EST_DASHBOARD_PROXY_SECRET"
