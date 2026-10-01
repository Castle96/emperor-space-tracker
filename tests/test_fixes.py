"""Tests for the correctness and safety fixes in the shared plumbing.

Grouped by module rather than by ticket: what matters here is that each of these
behaviours is now pinned, because every one of them was a silent wrong answer
rather than a crash.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from emperor_space_tracker.config import SAR_COLLECTIONS, SAR_RAW_ONLY, SarConfig, load_config
from emperor_space_tracker.errors import ConfigError, StoreError
from emperor_space_tracker.models import BreedingHabitat, utc_now
from emperor_space_tracker.net import require_http_url
from emperor_space_tracker.sources.sar import classify_cell, db
from emperor_space_tracker.store import Store

# --------------------------------------------------------------------------- #
# net: URL scheme allowlisting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url", [
    "https://example.invalid/a",
    "http://example.invalid/a",
    "https://example.invalid:8443/a?b=c#d",
])
def test_http_and_https_are_accepted(url: str) -> None:
    """The two schemes the project actually fetches stay usable."""
    assert require_http_url(url) == url


@pytest.mark.parametrize("url", [
    "file:///etc/shadow",
    "ftp://example.invalid/x",
    "gopher://example.invalid/",
    "data:text/plain,hi",
])
def test_other_schemes_are_refused(url: str) -> None:
    """Anything that is not HTTP(S) is rejected before a socket is opened.

    Without this, a value sourced from a colony catalogue or a config file could
    redirect a fetch at ``file://`` or ``gopher://`` and turn a read-only
    collector into a local-file reader.
    """
    with pytest.raises(ValueError, match="refusing to open non-HTTP URL scheme"):
        require_http_url(url)


@pytest.mark.parametrize("url", ["//example.invalid/x", "/relative/path", ""])
def test_urls_without_an_absolute_http_scheme_are_refused(url: str) -> None:
    """A scheme-relative or bare path is rejected too, with its own message."""
    with pytest.raises(ValueError):
        require_http_url(url)


# --------------------------------------------------------------------------- #
# sar: linear power to dB, and the polarisation/dataset mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("linear", "expected"), [
    (1.0, 0.0),
    (0.1, -10.0),
    (0.01, -20.0),
])
def test_db_converts_linear_power(linear: float, expected: float) -> None:
    """The GEE boundary conversion matches the textbook sigma0 relation."""
    assert db(linear) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_db_refuses_values_that_are_not_measurements(bad: float) -> None:
    """Zero and non-finite sigma0 are masked pixels, not measurements.

    ``log10`` of these raises or yields nonsense, and a nonsense dB value would
    land in the store and be read later as a real observation.
    """
    with pytest.raises(ValueError, match="strictly positive"):
        db(bad)


def test_db_matches_the_documented_backscatter_range() -> None:
    """A typical consolidated-ice sigma0 lands in the classified ice band."""
    assert classify_cell(db(0.4))[1] is False


def test_default_polarisation_resolves_to_a_collection_that_has_it() -> None:
    """The default band must exist in the collection the config selects.

    GRD products carry only VV and VH, so this asserts the shipped default is a
    band GRD actually contains rather than one that silently reduces to nothing.

    ``use_user_config=False`` because the claim is about the *shipped* default.
    Reading the developer's ~/.config would make this assert whatever they last
    edited, and pass or fail depending on whose machine ran it.
    """
    config = load_config(use_user_config=False)
    assert config.sar.polarisation.upper() == "VV"
    assert SAR_COLLECTIONS["VV"] == "COPERNICUS/S1_GRD"


def test_every_configurable_band_is_mapped_to_a_collection() -> None:
    """Config validation and collection selection cannot drift apart.

    Both read the same table, so a band accepted by ``config validate`` is
    guaranteed to resolve to a dataset rather than erroring at query time.
    """
    for band in ("VV", "VH", "HH", "HV"):
        assert SAR_COLLECTIONS[band].startswith("COPERNICUS/S1_")
    assert set(SAR_RAW_ONLY) == {"HH", "HV"}
    assert SAR_RAW_ONLY.isdisjoint({"VV", "VH"})


def test_config_rejects_a_band_no_dataset_provides(tmp_path: Path) -> None:
    """An unsupported polarisation fails validation rather than a live query."""
    path = tmp_path / "bad.toml"
    path.write_text('[sar]\npolarisation = "XX"\n')

    with pytest.raises(ConfigError, match="must be one of"):
        load_config(path, use_user_config=False)


def test_hh_resolves_to_the_raw_product_not_grd() -> None:
    """HH is only available from S1_RAW, which is the reason the mapping exists.

    GRD is dual-pol VV+VH. Requesting HH against it is not a slow query, it is
    an empty band selection, so HH must route to RAW.
    """
    assert SAR_COLLECTIONS["HH"] == "COPERNICUS/S1_RAW"
    assert "HH" in SAR_RAW_ONLY
    assert SarConfig(polarisation="HH").polarisation.upper() == "HH"


# --------------------------------------------------------------------------- #
# store: the v1 -> v2 column rename
# --------------------------------------------------------------------------- #


def _v1_database(path: str) -> None:
    """Create a database shaped like schema version 1, before the rename.

    Includes a v1-shaped ``colonies`` table with a row, because a real v1
    install had one and the v2->v3 step has to add its columns to it. A fixture
    that omitted the table would let an ``ALTER TABLE colonies`` step that
    crashes on a missing table pass unnoticed.
    """
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sar_cells ("
        "scene_id TEXT NOT NULL, row_index INTEGER NOT NULL, col_index INTEGER NOT NULL,"
        "sigma0_hh_db REAL NOT NULL, is_open_water INTEGER NOT NULL, classification TEXT NOT NULL,"
        "PRIMARY KEY (scene_id, row_index, col_index))"
    )
    conn.execute(
        "CREATE TABLE colonies ("
        "colony_id TEXT PRIMARY KEY, name TEXT NOT NULL, latitude REAL NOT NULL,"
        "longitude REAL NOT NULL, region TEXT NOT NULL, population_estimate INTEGER,"
        "population_year INTEGER, population_source TEXT NOT NULL, fast_ice_ratio REAL,"
        "last_census_at INTEGER, updated_at INTEGER NOT NULL,"
        "notes TEXT NOT NULL DEFAULT '')"
    )
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES('schema_version', '1')")
    conn.execute(
        "INSERT INTO colonies VALUES("
        "'cpe-old', 'Old Colony', -77.5, 166.6, 'Ross Sea', 4200, 2019, 'SCAR', 0.9,"
        "NULL, 1600000000000000, 'kept')"
    )
    conn.executemany(
        "INSERT INTO sar_cells VALUES(?,?,?,?,?,?)",
        [(f"S{i}", r, c, -7.5, 0, "consolidated_ice")
         for i in range(3) for r in range(2) for c in range(2)],
    )
    conn.commit()
    conn.close()


def test_existing_database_is_migrated_and_keeps_its_rows(tmp_path: Path) -> None:
    """Opening a pre-rename database renames the column and loses nothing.

    A renamed column breaks any database written by an earlier build, and this
    is the upgrade path a deployed install takes. Losing rows here would silently
    truncate the historical series the trend analysis depends on.
    """
    path = str(tmp_path / "tracker.sqlite3")
    _v1_database(path)

    store = Store(path)
    try:
        columns = [r[1] for r in store.conn.execute("PRAGMA table_info(sar_cells)")]
        assert "sigma0_db" in columns
        assert "sigma0_hh_db" not in columns
        assert store.conn.execute("SELECT count(*) FROM sar_cells").fetchone()[0] == 12
        assert store.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == "3"
    finally:
        store.close()


def test_v1_colony_rows_gain_a_species_and_are_kept(tmp_path: Path) -> None:
    """A v1 colony row survives the species migration as a fast-ice breeder.

    A v1 database cannot know better: every colony it held was an Emperor, so
    the backfill is the truth for that data rather than a guess. The row must
    come back out of the store as a usable ``Colony`` and not merely survive as
    loose columns.
    """
    path = str(tmp_path / "tracker.sqlite3")
    _v1_database(path)

    store = Store(path)
    try:
        colonies = store.colonies()
        assert len(colonies) == 1
        migrated = colonies[0]
        assert migrated.colony_id == "cpe-old"
        assert migrated.notes == "kept"
        assert migrated.population_estimate == 4200
        # v1 had nowhere to record a taxon, so the migration supplies the one
        # value that was true of every row it could have contained.
        assert migrated.species == "Aptenodytes forsteri"
        assert migrated.breeding_habitat is BreedingHabitat.FAST_ICE
        assert migrated.monitors_fast_ice
    finally:
        store.close()


def test_migration_is_not_applied_twice(tmp_path: Path) -> None:
    """Reopening a migrated database is a no-op rather than a duplicate rename."""
    path = str(tmp_path / "tracker.sqlite3")
    _v1_database(path)
    for _ in range(3):
        store = Store(path)
        store.close()
    assert sqlite3.connect(path).execute("SELECT count(*) FROM sar_cells").fetchone()[0] == 12


def test_a_future_schema_is_refused_rather_than_guessed_at(tmp_path: Path) -> None:
    """A database from a newer build is refused with a clear message.

    Opening it and writing to columns whose meaning has since changed would
    corrupt data the newer build owns.
    """
    path = str(tmp_path / "tracker.sqlite3")
    _v1_database(path)
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(StoreError, match="only understands up to version"):
        Store(path)


def test_a_schema_without_a_version_stamp_is_not_migrated(tmp_path: Path) -> None:
    """A current-shape database with no version row is left alone.

    A database built straight from the schema -- by a test, or by a tool that
    applies it without stamping -- already has the renamed column. Trusting the
    missing version as "1" would try to rename a column that is not there and
    refuse to open a perfectly valid store.
    """
    from emperor_space_tracker.store import _SCHEMA

    path = str(tmp_path / "unstamped.sqlite3")
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.close()

    store = Store(path)
    try:
        columns = [r[1] for r in store.conn.execute("PRAGMA table_info(sar_cells)")]
        assert "sigma0_db" in columns
        assert "sigma0_hh_db" not in columns
    finally:
        store.close()


def test_fresh_database_gets_the_current_schema_directly(tmp_path: Path) -> None:
    """A new file is built at the current version, not migrated into it."""
    store = Store(str(tmp_path / "new.sqlite3"))
    try:
        columns = [r[1] for r in store.conn.execute("PRAGMA table_info(sar_cells)")]
        assert "sigma0_db" in columns
        assert store.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == "3"
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# models: timezone handling that the stores depend on
# --------------------------------------------------------------------------- #


def test_utc_now_is_always_aware() -> None:
    """Naive datetimes are the cause of the out-of-order series bugs."""
    assert utc_now().tzinfo is not None
    assert math.isfinite(utc_now().timestamp())
