"""SQLite-backed observation store with bounded growth.

Design constraints, in priority order:

1. **The database must never fill an edge node's flash.** Every write path is
   paired with a retention policy; :meth:`Store.prune` enforces both an age
   window and a hard byte ceiling.
2. **A power cut must not corrupt history.** WAL journalling, ``synchronous=NORMAL``
   and a single-transaction write path mean a node that loses power mid-poll
   restarts with its last committed poll intact and nothing half-written.
3. **The dashboard is a separate process and may be reading while the daemon
   writes.** WAL gives the reader a consistent snapshot without blocking the
   writer, which is exactly the single-writer/many-reader case.
4. **Memory stays flat.** The page cache is pinned to a small, explicit size
   (:data:`_CACHE_KIB`) rather than SQLite's default, and every query is
   streaming or windowed. Nothing ever does ``SELECT *`` over an unbounded set.

Timestamps are stored as integer microseconds since the Unix epoch. That is
compact, sorts correctly with plain B-tree comparison, converts back exactly, and
avoids SQLite's date functions and their text-format ambiguities entirely.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, NamedTuple

from .errors import StoreError
from .models import (
    BreedingHabitat,
    Colony,
    FastIceCell,
    SarScene,
    Severity,
    SourceHealth,
    SpaceWeatherSnapshot,
    utc_now,
)

__all__ = [
    "AlertRecord",
    "Store",
    "StoreStats",
    "bulk_series",
    "from_micros",
    "has_column",
    "open_default_store",
    "to_micros",
]

_LOG = logging.getLogger("emperor.store")

#: Page cache ceiling, KiB. 2 MiB is ample for the write volume of a five-minute
#: poll and is the single most effective lever on resident memory for a
#: long-lived SQLite connection.
_CACHE_KIB: Final = 2048

_SCHEMA_VERSION: Final = 3


class _Migration(NamedTuple):
    """One forward schema step.

    A step is guarded by the *shape* it expects rather than by the recorded
    version alone, because a database can be built straight from ``_SCHEMA``
    without a version row ever being stamped -- by a test, or by a tool that
    applies the schema itself. Two independent preconditions describe that
    shape.

    Attributes
    ----------
    from_version
        The version this step upgrades a database *from*.
    requires
        ``(table, column)`` pairs that must all exist before the step runs.
        Used by a rename, which is only correct against the old shape.
    creates
        ``(table, column)`` pairs that must all be *absent* before the step
        runs. Used by ``ADD COLUMN``, which fails outright against a shape that
        already has them. If any one of these is present the step is skipped
        entirely, so a partially-applied step is a no-op rather than an error.
    statements
        The SQL to run, in order.
    backfill
        Optional SQL run after ``statements``. ``ALTER TABLE ... ADD COLUMN``
        only accepts a *constant* default, so a column whose correct value for
        existing rows is not that constant cannot be filled in the DDL. This is
        where the real value goes. Leaving it out of a backfill would leave
        pre-existing rows holding the DDL default, which for a taxon name is an
        empty string rather than an error.
    """

    from_version: int
    requires: tuple[tuple[str, str], ...]
    creates: tuple[tuple[str, str], ...]
    statements: tuple[str, ...]
    backfill: tuple[str, ...] = ()


#: Ordered schema migrations.
#:
#: A fresh database skips this table entirely: ``_SCHEMA`` already describes the
#: current shape, so there is nothing to step forward from. Only a database that
#: already holds tables is walked through these, which is what keeps a renamed
#: column from either crashing an upgraded install or, worse, being silently
#: ignored while the writer assumes the new name exists.
_MIGRATIONS: Final[tuple[_Migration, ...]] = (
    # The sigma0 column was named for the polarisation it was assumed to carry.
    # It now holds whichever band the scene records, so the band belongs in the
    # scene row rather than in the column name.
    _Migration(
        from_version=1,
        requires=(("sar_cells", "sigma0_hh_db"),),
        creates=(("sar_cells", "sigma0_db"),),
        statements=(
            "ALTER TABLE sar_cells RENAME COLUMN sigma0_hh_db TO sigma0_db",
        ),
    ),
    # A colony used to be implicitly an Emperor penguin, because the catalogue
    # only ever held Emperor colonies and the schema had nowhere to say
    # otherwise. `species` and `breeding_habitat` make the taxon explicit and
    # record whether fast-ice monitoring applies, which is what lets a
    # land-nesting species be catalogued without the fast-ice pipeline inventing
    # a reading for it.
    #
    # Existing rows backfill as Aptenodytes forsteri on fast ice, because that
    # is what every row a v2 database could hold was: the catalogue had one
    # species in it and the writer hardcoded it. `breeding_habitat` defaults to
    # 'fast_ice' in the DDL, so it needs no backfill; `species` cannot take a
    # non-constant default and needs an explicit UPDATE.
    _Migration(
        from_version=2,
        requires=(),
        creates=(("colonies", "species"), ("colonies", "breeding_habitat")),
        statements=(
            "ALTER TABLE colonies ADD COLUMN species TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE colonies ADD COLUMN breeding_habitat TEXT NOT NULL DEFAULT 'fast_ice'",
            "ALTER TABLE colonies ADD COLUMN common_name TEXT NOT NULL DEFAULT ''",
        ),
        backfill=(
            "UPDATE colonies SET species = 'Aptenodytes forsteri' "
            "WHERE species IS NULL OR species = ''",
            "UPDATE colonies SET common_name = 'Emperor penguin' "
            "WHERE common_name IS NULL OR common_name = ''",
        ),
    ),
)


def has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Report whether ``table`` has a column named ``column``.

    Used by the migration runner as a precondition, so a step only runs when the
    shape it is meant to change is actually still there.

    Parameters
    ----------
    conn
        An open connection.
    table
        Table to inspect.
    column
        Column to look for.

    Returns
    -------
    bool
        Whether the column exists.
    """
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    """Report whether ``table`` exists.

    ``PRAGMA table_info`` on a missing table returns no rows, which is
    indistinguishable from a table that exists but has no such column. A
    migration therefore needs both answers separately: one to decide the step
    is unnecessary because the column is already there, and one to skip the step
    entirely because ``_SCHEMA`` is about to create the whole table anyway.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _colony_from_row(row: sqlite3.Row) -> Colony:
    """Rehydrate a :class:`~emperor_space_tracker.models.Colony` from a row.

    The single place a stored row becomes a ``Colony``. It exists because that
    mapping was previously written out again at every read site, which meant
    adding a field to the model was a five-file edit and a missing one was a
    silent ``None`` rather than an error.
    """
    return Colony(
        colony_id=row["colony_id"],
        name=row["name"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        region=row["region"],
        population_estimate=row["population_estimate"],
        population_year=row["population_year"],
        population_source=row["population_source"],
        species=row["species"],
        breeding_habitat=BreedingHabitat(row["breeding_habitat"]),
        common_name=row["common_name"],
        fast_ice_ratio=row["fast_ice_ratio"],
        last_census_at=(from_micros(row["last_census_at"]) if row["last_census_at"] else None),
        notes=row["notes"],
    )


def to_micros(moment: datetime) -> int:
    """Convert a datetime to integer microseconds since the Unix epoch.

    Naive datetimes are interpreted as UTC rather than rejected, because the
    NOAA feeds occasionally omit an offset. Silently assuming local time here
    would put Antarctic winter observations hours out of position.

    Parameters
    ----------
    moment
        The datetime to convert.

    Returns
    -------
    int
        Microseconds since 1970-01-01T00:00:00Z.

    Examples
    --------
    >>> to_micros(datetime(1970, 1, 1, tzinfo=UTC))
    0
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp() * 1_000_000)


def from_micros(value: int) -> datetime:
    """Convert integer microseconds since the epoch back to an aware datetime.

    Parameters
    ----------
    value
        Microseconds since the Unix epoch.

    Returns
    -------
    datetime
        A timezone-aware UTC datetime.

    Examples
    --------
    >>> from_micros(0).isoformat()
    '1970-01-01T00:00:00+00:00'
    """
    return datetime.fromtimestamp(value / 1_000_000, tz=UTC)


def _iso(moment: datetime | None) -> str | None:
    """Render a datetime as an ISO-8601 string, or ``None``."""
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string back to an aware datetime, or ``None``."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per ingested space weather sample. Indexed by observation time
-- because every dashboard read is a time-window query.
CREATE TABLE IF NOT EXISTS plasma (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at       INTEGER NOT NULL,
    ingested_at       INTEGER NOT NULL,
    source            TEXT    NOT NULL,
    speed_kms         REAL,
    density_per_cm3   REAL,
    temperature_k     REAL,
    active            INTEGER NOT NULL,
    quality           INTEGER,
    provenance        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS plasma_observed_idx ON plasma(observed_at DESC);

CREATE TABLE IF NOT EXISTS magnetic_field (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at   INTEGER NOT NULL,
    ingested_at   INTEGER NOT NULL,
    source        TEXT    NOT NULL,
    bt_nt         REAL,
    bz_gsm_nt     REAL,
    by_gsm_nt     REAL,
    density_pct   REAL,
    active        INTEGER NOT NULL,
    provenance    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS magnetic_field_observed_idx ON magnetic_field(observed_at DESC);

CREATE TABLE IF NOT EXISTS kp_index (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at   INTEGER NOT NULL,
    ingested_at   INTEGER NOT NULL,
    kp_index      INTEGER,
    estimated_kp  REAL,
    provenance    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS kp_index_observed_idx ON kp_index(observed_at DESC);

CREATE TABLE IF NOT EXISTS f10_7_flux (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at INTEGER NOT NULL,
    ingested_at INTEGER NOT NULL,
    flux_sfu    REAL,
    provenance  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS f10_7_flux_observed_idx ON f10_7_flux(observed_at DESC);

-- Denormalised per-poll snapshot. One row per successful polling pass; this is
-- the table the dashboard's sparklines read, so it is a single row fetch
-- instead of a join.
CREATE TABLE IF NOT EXISTS space_weather (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at   INTEGER NOT NULL,
    ingested_at   INTEGER NOT NULL,
    speed_kms     REAL,
    density_per_cm3 REAL,
    bz_gsm_nt     REAL,
    bt_nt         REAL,
    kp_value      REAL,
    kp_is_estimate INTEGER NOT NULL,
    f107_sfu      REAL,
    storm_active  INTEGER NOT NULL DEFAULT 0,
    degraded      TEXT,
    provenance    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS space_weather_observed_idx ON space_weather(observed_at DESC);

CREATE TABLE IF NOT EXISTS colonies (
    colony_id         TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    latitude          REAL NOT NULL,
    longitude         REAL NOT NULL,
    region            TEXT NOT NULL,
    population_estimate INTEGER,
    population_year   INTEGER,
    population_source TEXT    NOT NULL,
    species           TEXT    NOT NULL,
    breeding_habitat  TEXT    NOT NULL,
    common_name       TEXT    NOT NULL DEFAULT '',
    fast_ice_ratio    REAL,
    last_census_at    INTEGER,
    updated_at        INTEGER NOT NULL,
    notes             TEXT    NOT NULL DEFAULT ''
);


CREATE TABLE IF NOT EXISTS sar_scenes (
    scene_id           TEXT PRIMARY KEY,
    observed_at        INTEGER NOT NULL,
    ingested_at        INTEGER NOT NULL,
    platform           TEXT NOT NULL,
    orbit_type         TEXT NOT NULL,
    polarisation       TEXT NOT NULL,
    incidence_angle_deg REAL,
    colony_id          TEXT NOT NULL,
    mean_db            REAL NOT NULL,
    min_db             REAL NOT NULL,
    max_db             REAL NOT NULL,
    std_db             REAL NOT NULL,
    frozen_ice_fraction REAL NOT NULL,
    open_water_fraction REAL NOT NULL,
    stability          TEXT NOT NULL,
    provenance         TEXT NOT NULL,
    bytes_processed    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS sar_scenes_observed_idx ON sar_scenes(observed_at DESC);
CREATE INDEX IF NOT EXISTS sar_scenes_colony_idx ON sar_scenes(colony_id, observed_at DESC);

-- The backscatter matrix. This is by far the largest table: a 41x41 grid is
-- 1681 rows per scene. It is stored relationally rather than as a blob so the
-- dashboard can window a heatmap slice without decoding an entire archive.
CREATE TABLE IF NOT EXISTS sar_cells (
    scene_id      TEXT    NOT NULL,
    row_index     INTEGER NOT NULL,
    col_index     INTEGER NOT NULL,
    sigma0_db  REAL    NOT NULL,
    is_open_water INTEGER NOT NULL,
    classification TEXT   NOT NULL,
    PRIMARY KEY (scene_id, row_index, col_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id     TEXT    NOT NULL,
    severity    TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    body        TEXT    NOT NULL,
    context     TEXT,
    fired_at    INTEGER NOT NULL,
    delivered   INTEGER NOT NULL DEFAULT 0,
    delivery_error TEXT
);
CREATE INDEX IF NOT EXISTS alerts_fired_idx ON alerts(fired_at DESC);
CREATE INDEX IF NOT EXISTS alerts_rule_idx ON alerts(rule_id, fired_at DESC);

CREATE TABLE IF NOT EXISTS source_health (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at  INTEGER NOT NULL,
    source       TEXT    NOT NULL,
    ok           INTEGER NOT NULL,
    status       TEXT    NOT NULL,
    latency_ms   INTEGER NOT NULL,
    record_count INTEGER NOT NULL,
    detail       TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS source_health_observed_idx ON source_health(observed_at DESC);

-- Alert rule latches live in the database, not in memory. A node that restarts
-- mid-storm must not forget it is already in a storm, or every restart would
-- re-page the duty operator.
CREATE TABLE IF NOT EXISTS rule_state (
    rule_id      TEXT PRIMARY KEY,
    latched      INTEGER NOT NULL DEFAULT 0,
    streak       INTEGER NOT NULL DEFAULT 0,
    last_fired   INTEGER,
    updated_at   INTEGER NOT NULL,
    context      TEXT
);
"""


@dataclass(frozen=True, slots=True)
class AlertRecord:
    """A fired alert, as returned by :meth:`Store.recent_alerts`.

    Attributes
    ----------
    rule_id
        Identifier of the rule that fired.
    severity
        Severity ladder value.
    title
        One-line summary.
    body
        Detail paragraph.
    context
        Decoded JSON mapping of the values that triggered it.
    fired_at
        When the rule fired.
    delivered
        Whether the webhook accepted it.
    delivery_error
        Delivery failure detail, if any.
    """

    rule_id: str
    severity: Severity
    title: str
    body: str
    context: dict[str, Any]
    fired_at: datetime
    delivered: bool
    delivery_error: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable rendering of the record."""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "title": self.title,
            "body": self.body,
            "context": self.context,
            "fired_at": _iso(self.fired_at),
            "delivered": self.delivered,
            "delivery_error": self.delivery_error,
        }


@dataclass(frozen=True, slots=True)
class StoreStats:
    """Row counts and on-disk size, for ``est status``.

    Attributes
    ----------
    table
        Table name.
    rows
        Row count.
    """

    table: str
    rows: int

    def __str__(self) -> str:
        """Return a right-aligned ``table  rows`` line."""
        return f"{self.table:<18} {self.rows:>9,}"


#: Tables subject to age-based retention. Order matters: children before parents.
_PRUNABLE: Final[tuple[tuple[str, str], ...]] = (
    ("sar_cells", "sar_scenes"),
    ("sar_scenes", "observed_at"),
    ("plasma", "observed_at"),
    ("magnetic_field", "observed_at"),
    ("kp_index", "observed_at"),
    ("f10_7_flux", "observed_at"),
    ("space_weather", "observed_at"),
    ("source_health", "observed_at"),
    ("alerts", "fired_at"),
)


class Store:
    """Durable observation store backed by a single SQLite file.

    Parameters
    ----------
    path
        Database file path. ``":memory:"`` is accepted for tests.
    max_bytes
        On-disk ceiling. Pruning stops at this size. ``0`` disables the
        byte-based policy and relies on the age window alone.

    Examples
    --------
    >>> store = Store(":memory:")
    >>> [s.table for s in store.stats()][:3]
    ['plasma', 'magnetic_field', 'kp_index']
    >>> store.close()
    """

    def __init__(self, path: str | Path, *, max_bytes: int = 0) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.path = str(path)
        self.max_bytes = max_bytes
        self._conn: sqlite3.Connection | None = None
        self._connect()

    # -- lifecycle ---------------------------------------------------------

    def _connect(self) -> None:
        """Open the connection and apply pragmas plus schema."""
        try:
            conn = sqlite3.connect(
                self.path,
                timeout=15.0,
                isolation_level=None,  # explicit transaction control
                check_same_thread=False,
            )
        except sqlite3.Error as exc:
            msg = f"cannot open database {self.path}: {exc}"
            raise StoreError(msg) from exc

        conn.row_factory = sqlite3.Row
        # WAL lets the dashboard read a consistent snapshot while the daemon
        # writes, which is precisely this workload's concurrency pattern.
        # NORMAL rather than FULL: WAL + NORMAL survives application crashes
        # and OS crashes, and only risks the last few milliseconds of writes on
        # a power cut. For a polling daemon rebuilding from live APIs every five
        # minutes, that trade is free.
        for pragma in (
            "PRAGMA journal_mode=WAL",
            f"PRAGMA cache_size=-{_CACHE_KIB}",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA foreign_keys=ON",
            "PRAGMA temp_store=MEMORY",
            "PRAGMA mmap_size=0",
        ):
            try:
                conn.execute(pragma)
            except sqlite3.Error as exc:  # pragma: no cover - platform dependent
                _LOG.debug("pragma %s failed: %s", pragma, exc)

        try:
            self._migrate(conn)
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
        except sqlite3.Error as exc:
            msg = f"cannot initialise schema in {self.path}: {exc}"
            raise StoreError(msg) from exc

        self._conn = conn
        _LOG.debug("opened store %s", self.path)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Bring an existing database up to the current schema version.

        Only databases that already hold tables are migrated. A new file gets
        the current schema directly from :data:`_SCHEMA`, and running a column
        rename against it would fail because the old column never existed there.

        The stored version is trusted only up to a point: a database carrying
        tables but no recorded version predates version tracking, and is assumed
        to be version 1 so the first migration still applies.

        Parameters
        ----------
        conn
            Open connection to an existing database.

        Raises
        ------
        StoreError
            If the database was written by a newer version of this package.
        """
        has_tables = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sar_cells'"
        ).fetchone()
        if has_tables is None:
            return

        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        current = int(row[0]) if row is not None else 1

        if current > _SCHEMA_VERSION:
            msg = (
                f"database schema is version {current}, but this build only understands "
                f"up to version {_SCHEMA_VERSION}; upgrade the package to open it"
            )
            raise StoreError(msg)

        for migration in _MIGRATIONS:
            if current > migration.from_version:
                continue
            touched = {t for t, _ in migration.requires} | {t for t, _ in migration.creates}
            absent_tables = sorted(t for t in touched if not _has_table(conn, t))
            missing = [f"{t}.{c}" for t, c in migration.requires if not has_column(conn, t, c)]
            present = [f"{t}.{c}" for t, c in migration.creates if has_column(conn, t, c)]
            if absent_tables or missing or present:
                _LOG.debug(
                    "schema migration from %d not needed "
                    "(absent_tables=%s missing=%s present=%s)",
                    migration.from_version, absent_tables, missing, present,
                )
                continue
            try:
                for statement in migration.statements:
                    conn.execute(statement)
                for statement in migration.backfill:
                    conn.execute(statement)
            except sqlite3.Error as exc:
                msg = (
                    f"schema migration from version {migration.from_version} "
                    f"failed: {exc}"
                )
                raise StoreError(msg) from exc
            _LOG.info(
                "applied schema migration from version %d", migration.from_version
            )

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the live connection, raising if the store is closed."""
        if self._conn is None:
            msg = f"store {self.path} is closed"
            raise StoreError(msg)
        return self._conn

    def close(self) -> None:
        """Checkpoint the WAL and close the connection."""
        if self._conn is None:
            return
        # A checkpoint can fail if the DB is locked or the fs is read-only;
        # that must not prevent close(), or the caller leaks the handle.
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        try:
            self._conn.close()
        finally:
            self._conn = None

    def __enter__(self) -> Store:
        """Enter a context that closes the store on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the store on context exit."""
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside a single immediate transaction.

        A poll writes to eight tables. Wrapping them in one transaction means
        the dashboard never observes a snapshot with a fresh Kp index and a
        three-poll-old plasma sample, and a failure part-way through rolls the
        whole pass back rather than leaving a half-ingested poll behind.

        Yields
        ------
        sqlite3.Connection
            The connection, with the transaction already begun.
        """
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # -- space weather -----------------------------------------------------

    def record_snapshot(
        self, snapshot: SpaceWeatherSnapshot, *, storm_kp: float = 5.0
    ) -> None:
        """Persist a whole space weather poll pass.

        Writes the component tables (plasma, magnetic field, Kp, F10.7) and the
        denormalised snapshot row inside one transaction.

        Parameters
        ----------
        snapshot
            The assembled snapshot. Individual components may be ``None`` if
            their upstream product failed; the pass is still recorded so the
            gap is visible on the dashboard rather than looking like quiet
            weather.
        storm_kp
            The node's configured storm threshold, used to set the
            ``storm_active`` flag the dashboard labels with. Supplied by the
            caller from ``[space_weather] storm_kp``: hardcoding 5.0 here meant
            the dashboard's storm badge disagreed with the operator's own
            configuration, so a node tuned to a stricter or looser threshold was
            labelled with a storm state its rules had never declared.
        """
        ingested = to_micros(utc_now())
        with self.transaction() as conn:
            plasma = snapshot.plasma
            if plasma is not None:
                conn.execute(
                    "INSERT INTO plasma(observed_at, ingested_at, source, speed_kms, "
                    "density_per_cm3, temperature_k, active, quality, provenance) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        to_micros(plasma.observed_at),
                        ingested,
                        plasma.source,
                        plasma.speed_kms,
                        plasma.density_per_cm3,
                        plasma.temperature_k,
                        int(plasma.active),
                        plasma.quality,
                        plasma.provenance,
                    ),
                )

            field_data = snapshot.magnetic_field
            if field_data is not None:
                conn.execute(
                    "INSERT INTO magnetic_field(observed_at, ingested_at, source, bt_nt, "
                    "bz_gsm_nt, by_gsm_nt, density_pct, active, provenance) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        to_micros(field_data.observed_at),
                        ingested,
                        field_data.source,
                        field_data.bt_nt,
                        field_data.bz_gsm_nt,
                        field_data.by_gsm_nt,
                        field_data.density_pct,
                        int(field_data.active),
                        field_data.provenance,
                    ),
                )

            kp = snapshot.kp
            if kp is not None:
                conn.execute(
                    "INSERT INTO kp_index("
                    "observed_at, ingested_at, kp_index, estimated_kp, provenance"
                    ") VALUES(?,?,?,?,?)",
                    (
                        to_micros(kp.observed_at),
                        ingested,
                        kp.kp_index,
                        kp.estimated_kp,
                        kp.provenance,
                    ),
                )

            if snapshot.f107_sfu is not None:
                conn.execute(
                    "INSERT INTO f10_7_flux(observed_at, ingested_at, flux_sfu, provenance) "
                    "VALUES(?,?,?,?)",
                    (
                        ingested,
                        ingested,
                        snapshot.f107_sfu,
                        "noaa.swpc.f107_cm_flux",
                    ),
                )

            conn.execute(
                "INSERT INTO space_weather(observed_at, ingested_at, speed_kms, "
                "density_per_cm3, bz_gsm_nt, bt_nt, kp_value, kp_is_estimate, f107_sfu, "
                "storm_active, degraded, provenance) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    to_micros(snapshot.observed_at),
                    ingested,
                    plasma.speed_kms if plasma else None,
                    plasma.density_per_cm3 if plasma else None,
                    field_data.bz_gsm_nt if field_data else None,
                    field_data.bt_nt if field_data else None,
                    kp.value if kp else None,
                    int(kp.is_estimated) if kp else 0,
                    snapshot.f107_sfu,
                    int(kp.storm_level(storm_kp) if kp else False),
                    json.dumps(list(snapshot.degraded)) if snapshot.degraded else None,
                    "noaa.swpc",
                ),
            )

    def space_weather_series(
        self,
        *,
        hours: int = 24,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        """Return space weather snapshots over a trailing time window.

        Parameters
        ----------
        hours
            Trailing window in hours.
        limit
            Maximum rows returned, newest first, applied after windowing.

        Returns
        -------
        list[dict[str, Any]]
            Rows with parsed ``observed_at`` datetimes, oldest first, which is
            the order every chart in the dashboard wants.
        """
        since = to_micros(utc_now() - timedelta(hours=hours))
        cursor = self.conn.execute(
            "SELECT * FROM space_weather WHERE observed_at >= ? "
            "ORDER BY observed_at DESC LIMIT ?",
            (since, limit),
        )
        rows = [self._snapshot_row(row) for row in cursor.fetchall()]
        rows.reverse()
        return rows

    @staticmethod
    def _snapshot_row(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a ``space_weather`` row to a dict with a real datetime."""
        record = dict(row)
        record["observed_at"] = from_micros(record["observed_at"])
        record["ingested_at"] = from_micros(record["ingested_at"])
        record["storm_active"] = bool(record["storm_active"])
        record["kp_is_estimate"] = bool(record["kp_is_estimate"])
        record["degraded"] = json.loads(record["degraded"]) if record["degraded"] else []
        return record

    def latest_space_weather(self) -> dict[str, Any] | None:
        """Return the most recent snapshot row, or ``None`` if empty.

        Returns
        -------
        dict[str, Any] | None
        """
        cursor = self.conn.execute(
            "SELECT * FROM space_weather ORDER BY observed_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return self._snapshot_row(row) if row else None

    def kp_series(self, *, hours: int = 72) -> list[dict[str, Any]]:
        """Return Kp samples over a trailing window, oldest first.

        Parameters
        ----------
        hours
            Trailing window in hours. Defaults to 72 so the 3-hour barometer
            cadence yields a useful storm-time series.

        Returns
        -------
        list[dict[str, Any]]
        """
        since = to_micros(utc_now() - timedelta(hours=hours))
        cursor = self.conn.execute(
            "SELECT observed_at, kp_index, estimated_kp FROM kp_index "
            "WHERE observed_at >= ? ORDER BY observed_at ASC",
            (since,),
        )
        return [
            {
                "observed_at": from_micros(row["observed_at"]),
                "kp_index": row["kp_index"],
                "estimated_kp": row["estimated_kp"],
                "value": (
                    row["estimated_kp"]
                    if row["estimated_kp"] is not None
                    else row["kp_index"]
                ),
            }
            for row in cursor
        ]

    # -- colonies ----------------------------------------------------------

    def upsert_colonies(self, colonies: Iterable[Colony]) -> int:
        """Insert or update colony records, preserving the primary key.

        Parameters
        ----------
        colonies
            Colonies to write.

        Returns
        -------
        int
            Number of rows written.
        """
        now = to_micros(utc_now())
        rows = [
            (
                c.colony_id,
                c.name,
                c.latitude,
                c.longitude,
                c.region,
                c.population_estimate,
                c.population_year,
                c.population_source,
                c.species,
                str(c.breeding_habitat),
                c.common_name,
                c.fast_ice_ratio,
                to_micros(c.last_census_at) if c.last_census_at else None,
                now,
                c.notes,
            )
            for c in colonies
        ]
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO colonies(colony_id, name, latitude, longitude, region, "
                "population_estimate, population_year, population_source, species, "
                "breeding_habitat, common_name, fast_ice_ratio, "
                "last_census_at, updated_at, notes) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(colony_id) DO UPDATE SET "
                "name=excluded.name, latitude=excluded.latitude, longitude=excluded.longitude, "
                "region=excluded.region, population_estimate=excluded.population_estimate, "
                "population_year=excluded.population_year, "
                "population_source=excluded.population_source, "
                "species=excluded.species, "
                "breeding_habitat=excluded.breeding_habitat, "
                "common_name=excluded.common_name, "
                "fast_ice_ratio=excluded.fast_ice_ratio, "
                "last_census_at=excluded.last_census_at, "
                "updated_at=excluded.updated_at, notes=excluded.notes",
                rows,
            )
        return len(rows)

    def colonies(self, *, limit: int = 50) -> list[Colony]:
        """Return known colonies, largest population first.

        Parameters
        ----------
        limit
            Maximum colonies returned.

        Returns
        -------
        list[Colony]
        """
        cursor = self.conn.execute(
            "SELECT * FROM colonies ORDER BY population_estimate DESC NULLS LAST, name ASC "
            "LIMIT ?",
            (limit,),
        )
        return [_colony_from_row(row) for row in cursor]

    def geojson_feature_collection(self) -> dict[str, Any]:
        """Return all colonies as a GeoJSON ``FeatureCollection``.

        Returns
        -------
        dict[str, Any]
            A map-ready collection the dashboard hands straight to Plotly.
        """
        return {
            "type": "FeatureCollection",
            "features": [c.as_geojson() for c in self.colonies()],
        }

    # -- SAR ---------------------------------------------------------------

    def record_sar_scene(self, scene: SarScene) -> int:
        """Persist a SAR scene and its full backscatter grid.

        The grid is inserted with a single ``executemany`` inside the same
        transaction as the scene header, so a partially-written matrix can
        never be observed by the dashboard.

        Parameters
        ----------
        scene
            The scene to store.

        Returns
        -------
        int
            Number of grid cells written.
        """
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sar_scenes(scene_id, observed_at, ingested_at, "
                "platform, orbit_type, polarisation, incidence_angle_deg, colony_id, "
                "mean_db, min_db, max_db, std_db, frozen_ice_fraction, "
                "open_water_fraction, stability, provenance, bytes_processed) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    scene.scene_id,
                    to_micros(scene.observed_at),
                    to_micros(utc_now()),
                    scene.platform,
                    scene.orbit_type,
                    scene.polarisation,
                    scene.incidence_angle_deg,
                    scene.colony_id,
                    scene.mean_db,
                    scene.min_db,
                    scene.max_db,
                    scene.std_db,
                    scene.frozen_ice_fraction,
                    scene.open_water_fraction,
                    scene.classify(),
                    scene.provenance,
                    scene.bytes_processed,
                ),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO sar_cells(scene_id, row_index, col_index, "
                "sigma0_db, is_open_water, classification) VALUES(?,?,?,?,?,?)",
                [
                    (
                    c.scene_id,
                    c.row,
                    c.col,
                    c.sigma0_db,
                    int(c.is_open_water),
                    c.classification,
                )
                    for c in scene.cells
                ],
            )
        return len(scene.cells)

    def recent_sar_scenes(
        self,
        *,
        limit: int = 20,
        colony_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return scene metadata rows, newest first.

        Parameters
        ----------
        limit
            Maximum scenes.
        colony_id
            Restrict to one colony.

        Returns
        -------
        list[dict[str, Any]]
            Metadata only; the grid is fetched separately by
            :meth:`sar_matrix`.
        """
        sql = "SELECT * FROM sar_scenes"
        params: list[Any] = []
        if colony_id is not None:
            sql += " WHERE colony_id = ?"
            params.append(colony_id)
        sql += " ORDER BY observed_at DESC LIMIT ?"
        params.append(limit)
        return [
            {**dict(row), "observed_at": from_micros(row["observed_at"]),
             "ingested_at": from_micros(row["ingested_at"]),
             "is_synthetic": str(row["provenance"]).startswith("synthetic")}
            for row in self.conn.execute(sql, params)
        ]

    def sar_matrix(self, scene_id: str) -> list[FastIceCell]:
        """Return the backscatter grid for one scene, row-major.

        Parameters
        ----------
        scene_id
            The scene identifier.

        Returns
        -------
        list[FastIceCell]
            Cells in row-major order. Empty if the scene is unknown.
        """
        cursor = self.conn.execute(
            "SELECT row_index, col_index, sigma0_db, is_open_water, classification "
            "FROM sar_cells WHERE scene_id = ? ORDER BY row_index ASC, col_index ASC",
            (scene_id,),
        )
        return [
            FastIceCell(
                scene_id=scene_id,
                row=row["row_index"],
                col=row["col_index"],
                sigma0_db=row["sigma0_db"],
                is_open_water=bool(row["is_open_water"]),
                classification=row["classification"],
            )
            for row in cursor
        ]

    def sigma0_trend(self, colony_id: str, *, limit: int = 12) -> list[dict[str, Any]]:
        """Return scene-level sigma0 statistics over time for one colony.

        This is the series the fast-ice stability trend chart plots: a sustained
        fall in mean backscatter is the signature of a widening polynya.

        Parameters
        ----------
        colony_id
            Colony to report on.
        limit
            Maximum scenes, newest first.

        Returns
        -------
        list[dict[str, Any]]
        """
        cursor = self.conn.execute(
            "SELECT scene_id, observed_at, mean_db, min_db, max_db, std_db, "
            "open_water_fraction, stability, provenance FROM sar_scenes "
            "WHERE colony_id = ? ORDER BY observed_at DESC LIMIT ?",
            (colony_id, limit),
        )
        rows = [
            {
                "scene_id": row["scene_id"],
                "observed_at": from_micros(row["observed_at"]),
                "mean_db": row["mean_db"],
                "min_db": row["min_db"],
                "max_db": row["max_db"],
                "std_db": row["std_db"],
                "open_water_fraction": row["open_water_fraction"],
                "stability": row["stability"],
                "provenance": row["provenance"],
            }
            for row in cursor
        ]
        rows.reverse()
        return rows

    # -- health ------------------------------------------------------------

    def record_health(self, health: Iterable[SourceHealth]) -> int:
        """Persist per-source health outcomes for a poll.

        Parameters
        ----------
        health
            One entry per attempted source.

        Returns
        -------
        int
            Rows written.
        """
        rows = [
            (
                to_micros(h.observed_at),
                h.source,
                int(h.ok),
                h.status,
                h.latency_ms,
                h.record_count,
                h.detail,
            )
            for h in health
        ]
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO source_health(observed_at, source, ok, status, latency_ms, "
                "record_count, detail) VALUES(?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def health_summary(self, *, minutes: int = 60) -> list[dict[str, Any]]:
        """Return the most recent health row per source within a window.

        Parameters
        ----------
        minutes
            Trailing window in minutes.

        Returns
        -------
        list[dict[str, Any]]
        """
        since = to_micros(utc_now() - timedelta(minutes=minutes))
        cursor = self.conn.execute(
            "SELECT h.* FROM source_health h "
            "JOIN (SELECT source, MAX(observed_at) AS latest FROM source_health "
            "      WHERE observed_at >= ? GROUP BY source) newest "
            "ON h.source = newest.source AND h.observed_at = newest.latest "
            "ORDER BY h.source",
            (since,),
        )
        return [
            {
                "source": row["source"],
                "ok": bool(row["ok"]),
                "status": row["status"],
                "latency_ms": row["latency_ms"],
                "record_count": row["record_count"],
                "detail": row["detail"],
                "observed_at": from_micros(row["observed_at"]),
            }
            for row in cursor
        ]

    # -- alerts ------------------------------------------------------------

    def record_alert(
        self,
        *,
        rule_id: str,
        severity: Severity,
        title: str,
        body: str,
        context: dict[str, Any],
        delivered: bool,
        delivery_error: str | None = None,
    ) -> None:
        """Persist a fired alert and its delivery outcome.

        Parameters
        ----------
        rule_id
            Rule identifier.
        severity
            Severity of the alert.
        title
            One-line summary.
        body
            Detail paragraph.
        context
            Values that triggered the alert.
        delivered
            Whether the channel accepted it.
        delivery_error
            Failure detail if delivery failed.
        """
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO alerts(rule_id, severity, title, body, context, fired_at, "
                "delivered, delivery_error) VALUES(?,?,?,?,?,?,?,?)",
                (
                    rule_id,
                    severity.value,
                    title,
                    body,
                    json.dumps(context, default=str),
                    to_micros(utc_now()),
                    int(delivered),
                    delivery_error,
                ),
            )

    def recent_alerts(self, *, limit: int = 25) -> list[AlertRecord]:
        """Return the most recent alerts, newest first.

        Parameters
        ----------
        limit
            Maximum alerts.

        Returns
        -------
        list[AlertRecord]
        """
        cursor = self.conn.execute(
            "SELECT * FROM alerts ORDER BY fired_at DESC LIMIT ?", (limit,)
        )
        return [
            AlertRecord(
                rule_id=row["rule_id"],
                severity=Severity(row["severity"]),
                title=row["title"],
                body=row["body"],
                context=json.loads(row["context"]) if row["context"] else {},
                fired_at=from_micros(row["fired_at"]),
                delivered=bool(row["delivered"]),
                delivery_error=row["delivery_error"],
            )
            for row in cursor
        ]

    def last_alert_at(self, rule_id: str) -> datetime | None:
        """Return when ``rule_id`` last fired, for cooldown evaluation.

        Parameters
        ----------
        rule_id
            Rule identifier.

        Returns
        -------
        datetime | None
            Firing time, or ``None`` if the rule has never fired.
        """
        row = self.conn.execute(
            "SELECT fired_at FROM alerts WHERE rule_id = ? ORDER BY fired_at DESC LIMIT 1",
            (rule_id,),
        ).fetchone()
        return from_micros(row["fired_at"]) if row else None

    def set_rule_state(
        self,
        rule_id: str,
        *,
        latched: bool,
        streak: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Persist a rule's latch and confirmation streak.

        This is deliberately on disk. During polar night a node may reboot on a
        brownout mid-storm; if the latch lived in memory the node would come
        back unaware it had already paged the operator.

        Parameters
        ----------
        rule_id
            Rule identifier.
        latched
            Whether the condition is currently active.
        streak
            Consecutive confirming samples.
        context
            Values the rule last evaluated.
        """
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO rule_state(rule_id, latched, streak, last_fired, updated_at, context) "
                "VALUES(?,?,?,"
                "COALESCE((SELECT MAX(fired_at) FROM alerts WHERE rule_id = ?), NULL), ?, ?) "
                "ON CONFLICT(rule_id) DO UPDATE SET latched=excluded.latched, "
                "streak=excluded.streak, last_fired=excluded.last_fired, "
                "updated_at=excluded.updated_at, context=excluded.context",
                (
                    rule_id,
                    int(latched),
                    streak,
                    rule_id,
                    to_micros(utc_now()),
                    json.dumps(context, default=str) if context else None,
                ),
            )

    def get_rule_state(self, rule_id: str) -> tuple[bool, int, dict[str, Any]] | None:
        """Return a rule's persisted latch, streak and context.

        Parameters
        ----------
        rule_id
            Rule identifier.

        Returns
        -------
        tuple[bool, int, dict[str, Any]] | None
            ``(latched, streak, context)``, or ``None`` if the rule is unknown.
        """
        row = self.conn.execute(
            "SELECT latched, streak, context FROM rule_state WHERE rule_id = ?", (rule_id,)
        ).fetchone()
        if row is None:
            return None
        return (
            bool(row["latched"]),
            int(row["streak"]),
            json.loads(row["context"]) if row["context"] else {},
        )

    # -- maintenance -------------------------------------------------------

    def prune(self, *, retention_days: int = 30) -> dict[str, int]:
        """Delete rows outside the retention window and enforce the size cap.

        Age-based pruning runs first. If the database is still over
        :attr:`max_bytes` afterwards, scenes are dropped oldest-first until it
        fits. The :class:`~emperor_space_tracker.models.Colony` table and the
        rule-state table are never pruned: the census is refreshed on its own
        schedule and the latches are operational state, not history.

        Parameters
        ----------
        retention_days
            Rows older than this are removed. ``<= 0`` disables age pruning.

        Returns
        -------
        dict[str, int]
            Rows deleted per table, plus a ``freed_bytes`` figure.

        Notes
        -----
        ``freed_bytes`` is the *measured* change in the on-disk size of the
        database and its WAL across the prune, so it is a lower bound on the
        space reclaimed: SQLite keeps deleted pages in the freelist and returns
        them to the filesystem only on ``VACUUM``, which is deliberately not run
        on every prune. A non-zero value here means pages were genuinely
        released; a zero after a large delete means they are still reusable but
        still held.
        """
        deleted: dict[str, int] = {}
        size_before = self.size_bytes()

        if retention_days > 0:
            cutoff = to_micros(utc_now() - timedelta(days=retention_days))
            with self.transaction() as conn:
                for table, column in _PRUNABLE:
                    if table == "sar_cells":
                        cursor = conn.execute(
                            "DELETE FROM sar_cells WHERE scene_id IN "
                            "(SELECT scene_id FROM sar_scenes WHERE observed_at < ?)",
                            (cutoff,),
                        )
                    else:
                        cursor = conn.execute(
                            # fixed table list
                            f"DELETE FROM {table} WHERE {column} < ?",
                            (cutoff,),
                        )
                    if cursor.rowcount > 0:
                        deleted[table] = cursor.rowcount

        self.vacuum_if_needed()
        if self.max_bytes:
            deleted.update(self._enforce_size_cap())

        deleted["freed_bytes"] = max(0, size_before - self.size_bytes())
        return deleted

    def _enforce_size_cap(self) -> dict[str, int]:
        """Drop oldest scenes until the database fits under the byte cap.

        Returns
        -------
        dict[str, int]
            ``{"scenes_dropped": n}`` if anything was dropped, else ``{}``.
        """
        size = self.size_bytes()
        if size <= self.max_bytes:
            return {}

        conn = self.conn
        dropped = 0
        with self.transaction():
            # Batch-delete oldest scenes; a single unbounded DELETE of the whole
            # grid would spike memory and hold the write lock for too long.
            for _ in range(200):
                rows = conn.execute(
                    "SELECT scene_id FROM sar_scenes ORDER BY observed_at ASC LIMIT 20"
                ).fetchall()
                if not rows:
                    break
                ids = [(row["scene_id"],) for row in rows]
                conn.executemany("DELETE FROM sar_cells WHERE scene_id = ?", ids)
                conn.executemany("DELETE FROM sar_scenes WHERE scene_id = ?", ids)
                dropped += len(ids)
                if self.size_bytes() <= self.max_bytes:
                    break

        if dropped:
            _LOG.warning(
                "size cap %d bytes exceeded; dropped %d oldest SAR scene(s)",
                self.max_bytes,
                dropped,
            )
            self.vacuum_if_needed()
        return {"scenes_dropped": dropped} if dropped else {}

    def vacuum_if_needed(self, *, threshold: float = 0.25) -> bool:
        """Run ``VACUUM`` only if reclaimable space is worth the rewrite.

        ``VACUUM`` rewrites the whole database, so doing it on every prune on a
        flash-based node would dominate the node's entire write endurance
        budget. This is gated on the free-page ratio.

        Parameters
        ----------
        threshold
            Minimum fraction of the file that must be free pages to act.

        Returns
        -------
        bool
            ``True`` if a vacuum was performed.
        """
        if self.path == ":memory:":
            return False
        try:
            page_count = self.conn.execute("PRAGMA page_count").fetchone()[0]
            free_pages = self.conn.execute("PRAGMA freelist_count").fetchone()[0]
        except sqlite3.Error:
            return False
        if page_count <= 0 or free_pages / page_count < threshold:
            return False
        _LOG.info("compacting database: %d/%d pages free", free_pages, page_count)
        try:
            self.conn.execute("VACUUM")
        except sqlite3.Error as exc:  # pragma: no cover - disk dependent
            _LOG.warning("VACUUM failed: %s", exc)
            return False
        return True

    def size_bytes(self) -> int:
        """Return the total on-disk size of the database and its WAL.

        Returns
        -------
        int
            Size in bytes, or ``0`` for an in-memory database.
        """
        if self.path == ":memory:":
            return 0
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(self.path + suffix)
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
        return total

    def stats(self) -> list[StoreStats]:
        """Return row counts for every tracked table.

        Returns
        -------
        list[StoreStats]
        """
        tables = (
            "plasma", "magnetic_field", "kp_index", "f10_7_flux", "space_weather",
            "colonies", "sar_scenes", "sar_cells", "alerts", "source_health", "rule_state",
        )
        results: list[StoreStats] = []
        for table in tables:
            try:
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                count = 0
            results.append(StoreStats(table=table, rows=int(count)))
        return results

    def purge(self) -> None:
        """Delete every observation row, keeping schema and rule state.

        Backs ``est reset --data``.
        """
        with self.transaction() as conn:
            for table, _ in _PRUNABLE:
                # table names come from the _PRUNABLE literal, not input
                with contextlib.suppress(sqlite3.Error):
                    conn.execute(f"DELETE FROM {table}")
        self.vacuum_if_needed(threshold=0.1)


def open_default_store(config: Any) -> Store:
    """Open the store described by a :class:`~emperor_space_tracker.config.Config`.

    Parameters
    ----------
    config
        A loaded configuration object.

    Returns
    -------
    Store
        An opened store with the size cap applied.
    """
    state_dir = config.paths.ensure()
    return Store(
        state_dir / "tracker.sqlite3",
        max_bytes=config.paths.max_db_mib * 1024 * 1024,
    )


def bulk_series(rows: Sequence[sqlite3.Row], *, key: str) -> dict[str, Any]:
    """Zip a list of rows into a column-oriented mapping for charting.

    Parameters
    ----------
    rows
        Rows to transpose.
    key
        Column whose values become the mapping keys.

    Returns
    -------
    dict[str, Any]
        ``{column_name: [values...]}`` in row order.
    """
    if not rows:
        return {}
    return {column: [row[column] for row in rows] for column in rows[0] if column != key}
