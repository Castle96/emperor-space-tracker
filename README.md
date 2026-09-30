# Emperor Space Tracker

An edge monitoring node for Antarctic fast ice during polar night. It fuses four
independent public data sources into one SQLite store, applies threshold rules
with hysteresis, and can post alerts to Discord.

| Domain | Source | Auth |
| --- | --- | --- |
| Solar wind, IMF, planetary K, F10.7 | [NOAA SWPC](https://services.swpc.noaa.gov/products) | none |
| Fast-ice backscatter | Sentinel-1 via Google Earth Engine | `earthengine authenticate` |
| Penguin colony census | SCAR / CCAMLR published counts (bundled) | none |
| Presence observations | [GBIF](https://www.gbif.org) occurrence search | none |
| Alert delivery | Discord webhook | `EMPEROR_DISCORD_WEBHOOK` |

## Species and breeding habitat

Every colony record names a `species` and a `breeding_habitat`, and those two
fields decide which of this node's channels are valid for it. A record whose
species is not tracked produces no data rather than misleading data.

| Species | Habitat | Fast-ice SAR | `fast_ice_ratio` |
| --- | --- | --- | --- |
| *Aptenodytes forsteri* (Emperor) | `fast_ice` | applies | meaningful |
| *Pygoscelis kerguelensis* (southeastern gentoo) | `land` | **skipped** | absent |

The Emperor is the only penguin in the world that breeds on sea ice, and it is
the only species the fast-ice pipeline exists for. A land-nesting species is
catalogued and reported on, but the SAR client declines to image it and the
fast-ice alert ignores it: a sigma0 measured over a beach describes the nearby
water, and every classification band downstream is calibrated for consolidated
sea ice.

`colonies.species` selects which species is in range, scored and paged. The
catalogue holds both; the setting is not a filter on the catalogue.

```toml
[colonies]
species = "Pygoscelis kerguelensis"   # Kerguelen gentoo, land breeder
max_distance_km = 400.0
include_gbif = true
```

The daemon core has **no third-party dependencies**. It runs on the Python
standard library alone, so a field node provisions on Arch, Ubuntu, Debian, RHEL
or Alpine with no compiler and no dependency tree to audit. Streamlit/Plotly and
`earthengine-api` are optional extras, and neither is ever imported by the
service process.

## Architecture

The system is a single Python process that runs on a schedule, fetches from live
APIs, writes to a local SQLite database, evaluates rules, and optionally posts to
Discord. A separate Streamlit process reads the same database and renders the
frontend.

```
                     ┌─────────────────────────────┐
                     │        CLI / systemd         │
                     │   est poll | est run | ...   │
                     └──────────────┬──────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │       PollEngine            │
                     │  one pass: fetch → persist  │
                     │        → evaluate → alert    │
                     └───┬──────────┬──────────────┘
                         │          │
            ┌───────────▼──┐  ┌────▼────────────────┐
            │  Sources      │  │   Alerts             │
            │  · swpc       │  │   · rules.py engine  │
            │  · biological │  │   · notifier.py disc │
            │  · sar        │  │   · NullNotifier      │
            └───────────────┘  └──────────────────────┘
                         │
            ┌───────────▼─────────────────────────────────────┐
            │              Store (SQLite, WAL)                  │
            │  plasma  mag  kp  f107  space_weather            │
            │  sar_scenes  sar_cells  colonies  source_health  │
            │  alerts  rule_state  meta                        │
            └───────────────────────────────────────────────────┘
                                  ▲
                                  │ same file, reader/writer
            ┌─────────────────────┴───────────────────────────┐
            │              Streamlit Dashboard                  │
            │  overview · space weather · fast ice · ...       │
            └───────────────────────────────────────────────────┘
```

### Why standard-library-only

A field node may sit unattended in an unheated hut with no compiler, no wheel
cache and no network to resolve transitive dependencies. The daemon core
(`config`, `net`, `store`, `sources`, `alerts`, `engine`) imports nothing beyond
the standard library. The two optional extras — `earthengine-api` for real SAR
and `streamlit`+`plotly` for the dashboard — are loaded only by the commands
that need them, never by the systemd unit.

### Data flow in one polling pass

1. **Space weather.** The SWPC client fetches four independent products —
   RTSW plasma, RTSW magnetic field, planetary K index and F10.7 flux — each
   from its own endpoint. Plasma and magnetic field are newest-first JSON arrays
   served with `Accept-Ranges: bytes`; the client requests a 32 KiB prefix and
   parses only the first complete records, so a 2.5 MB feed costs 32 KiB on the
   wire. Products are fetched independently so one broken feed degrades the
   snapshot instead of failing the whole pass.

2. **Colonies.** The biological client loads the bundled curated catalogue of
   13 colonies across two species, optionally queries GBIF occurrence search for
   presence evidence (pages are aggregated and discarded one at a time to bound
   memory), and filters to colonies of the configured species within the
   configured radius of the site. The GBIF search box is derived from the
   coordinates of the catalogue records for that species, so it cannot
   disagree with the data; there is no hardcoded country or latitude filter to
   fall out of step with a taxonomic change.

3. **SAR.** If fast-ice-breeding colonies are in range and the SAR backend is
   available, the SAR client produces one scene per colony. Land-nesting
   colonies are skipped and reported as skipped in the source-health detail. The synthetic backend runs a
   physics-based simulator (consolidated-ice baseline, orographic wind
   roughening, pressure ridges, a chamfer-distance lead network, seasonal breakup
   model). The GEE backend filters `COPERNICUS/S1_GRD` or `COPERNICUS/S1_RAW`
   by colony footprint, date and orbit pass, then reduces each scene to a
   `grid_cells × grid_cells` sigma0 matrix with `reduceRegion`. The configured
   polarisation selects the collection: VV/VH read GRD, HH/HV read RAW.

4. **Persist.** Each write path commits independently inside the store: space
   weather snapshot (with component tables), colonies (upsert), SAR scenes and
   their 1681-cell grids, source health, and any fired alerts. One poll = one
   transaction per write path so a failure in one does not lose the others.

5. **Evaluate.** The rule engine reads the snapshot, the scenes and the
   colonies, runs each rule through the predicate, and applies confirmation
   streaks, hysteresis, cooldown and the persistent latch before firing. Alerts
   go to the configured dispatcher — Discord webhook or, by default, the null
   notifier that logs locally.

6. **Reclaim.** Between passes the engine forces a GC and, on glibc, calls
   `malloc_trim(0)` to return freed arenas to the kernel.

### Persistence and retention

SQLite is opened with WAL journalling, `synchronous=NORMAL`, an explicit 2 MiB
page cache, and `foreign_keys=ON`. A single-writer/many-reader pattern — daemon
writes, Streamlit reads — is the exact case WAL was designed for.

The store never grows without bound. An hourly prune removes rows older than
`paths.retention_days`, then drops oldest SAR scenes (and their `sar_cells`
rows, children before parents) until the on-disk size is under `paths.max_db_mib`.
`VACUUM` runs only when the freelist ratio crosses a threshold, because a full
rewrite on flash every prune would consume the node's write endurance.

Timestamps are integer microseconds since the Unix epoch. That is compact,
sorts correctly with plain B-tree comparison, converts back exactly, and avoids
SQLite's text-date ambiguities entirely.

Migrations are guarded by the schema *shape* they expect rather than by the
recorded version alone, because a database can be built straight from the
current schema without a version row ever being stamped. Each step declares the
columns it requires and the columns it creates; the step runs only if every
required column exists and no created column does.

### Alerting semantics

A naive `if kp >= 5: alert()` pages once per five-minute cycle for the duration
of a storm — 72 to 144 identical pages. Four mechanisms prevent that:

- **Confirmation streak.** A condition must hold for N consecutive polls before
  it fires. A one-minute Kp spike is not a storm.
- **Hysteresis.** Separate enter and clear thresholds. With
  `storm_kp = 5.0` and `storm_kp_clear = 4.0`, Kp oscillating across 4.5
  produces one transition, not fifty.
- **Cooldown.** A genuine re-entry will not re-page within the cooldown window.
- **Retry class.** Discord rate limits and 5xx responses are retried; a 4xx
  rejection is not. A server blip mid-storm used to discard the page, which
  from the operator's side is indistinguishable from a revoked webhook.
- **Persistent latch.** Whether a condition is active lives in SQLite, not
  memory. A node that browns out mid-storm and restarts knowing it is already in
  a storm pages once, not twice.

State transitions are reported explicitly — "storm began", "still storming",
"storm ended" — because the all-clear is the single most important message the
system can send.

### Daemon supervision

The `Daemon` loop sleeps in one-second slices so a `SIGTERM` from
`systemctl stop` is honoured within one second rather than up to five minutes,
which would force systemd to escalate to SIGKILL and discard the in-flight poll.
A failure counter pages the duty operator via a `node-unhealthy` alert after
`daemon.consecutive_failures_before_page` consecutive failed passes. A heartbeat
line is emitted hourly regardless of outcome.

### Dashboard

The Streamlit frontend is a thin read layer. Every figure is a query against the
same SQLite store the daemon writes; the dashboard holds no analysis of its own.
Thresholds live in the alert rules, classifications in the SAR backend, and
presence/absence semantics in the biological source. Duplicating any of that in a
view layer is how a dashboard ends up disagreeing with the daemon that produced
the data.

The dashboard module is import-safe without the extras installed, so `est doctor`
and `est --help` keep working on a headless node that has no Streamlit at all.

## Install

```sh
git clone https://github.com/kcastle96/emperor-space-tracker
cd emperor-space-tracker
uv sync                      # daemon only
uv sync --extra dashboard    # add the web UI
uv sync --extra sar          # add Earth Engine
```

Requires Python 3.13+. Any PEP 668-compliant installer works if you prefer not to
use `uv`:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e .                      # daemon
pip install -e '.[dashboard,sar]'     # everything
```

## Quick start

```sh
uv run est doctor        # verify environment, TLS and connectivity
uv run est poll          # one polling pass
uv run est status        # latest observations and node health
uv run est seed          # backfill synthetic history for the dashboard
uv run est dashboard     # launch the UI on :8501
```

`est seed` fills the store with plausible history so the dashboard has something
to draw. It is labelled `synthetic.sar-simulator/v1` throughout and is never
mixed with real observations.

## Deploying as a service

```sh
uv run est install                  # writes and enables the systemd user unit
systemctl --user status emperor-space-tracker.service
journalctl --user -u emperor-space-tracker.service -f
```

This is a **user** unit: no root, no login required, and the whole service lives
under your own account so a node can be reimaged without a root inventory. The
generated unit applies a filesystem sandbox (`ProtectSystem=strict`, a single
`ReadWritePaths` for the state directory), drops every address family except
IPv4/IPv6, and caps memory with a cgroup limit derived from
`daemon.max_rss_bytes`.

`est install` is idempotent. Re-run it after changing the config file or the
resource limits.

> **Watchdog is off deliberately.** A non-zero `WatchdogSec` makes systemd expect
> `sd_notify(WATCHDOG=1)` pings and kill the unit when they stop arriving. This
> daemon does not speak the notify protocol, so enabling it would abort a healthy
> process on a timer. Liveness is covered by `Restart=always` and `est doctor`.

## Configuration

Three layers, each overriding the last:

1. the packaged defaults (`src/emperor_space_tracker/data/default.toml`)
2. `~/.config/emperor-space-tracker/config.toml`
3. `EST_*` environment variables

```sh
uv run est config init --out ./my.toml   # write a documented starter file
uv run est config show                   # effective configuration
uv run est config validate               # fail loudly on a bad value
```

Global flags work on either side of the subcommand, so `est -v poll` and
`est poll -v` are the same thing.

### Sentinel-1 polarisation

`sar.polarisation` selects both the band and the Earth Engine product, because
the two are not independent:

| Polarisation | Collection | State |
| --- | --- | --- |
| `VV`, `VH` | `COPERNICUS/S1_GRD` | calibrated, orbit-corrected, speckle-filtered |
| `HH`, `HV` | `COPERNICUS/S1_RAW` | needs thermal-noise removal, calibration and speckle filtering before its dB values mean anything |

The GRD products are dual-pol **VV + VH** and contain no HH at all. Requesting HH
against GRD is not a slow query, it is an empty band selection, which is why the
default is `VV` and why the collection is derived from the setting rather than
hardcoded. VV is also the conventional channel for telling consolidated fast ice
from open water and leads: the volume-scattering contrast that separates intact
ice from briny nilas and open water is strongest in VV.

```toml
[sar]
backend = "gee"
polarisation = "VV"
```

## What the numbers mean

**Fast ice reads bright, open water reads dark.** Consolidated fast ice returns
roughly −8 to −2 dB because brine channels and pressure ridges act as volume
scatterers; open water and thin nilas return roughly −20 to −12 dB. A *drop* in
sigma0 between acquisitions is the standard remote-sensing signature of a lead
opening or the fast ice detaching from the coast.

**Presence is not abundance.** GBIF occurrences are observer effort. A grid cell
with 40 records means 40 observations were made there, not that 40 penguins are
present. Colonies come from published censuses, and each record carries its
`population_source` and year. Coordinates for threatened taxa are generalised by
GBIF, so summaries report `median_uncertainty_m` and
`coordinates_are_generalised`.

**Synthetic data is always labelled.** Every record carries a `provenance` field.
Synthetic scenes use `synthetic.sar-simulator/v1`; the dashboard shows a warning
when the selected colony is backed by them.

## Memory

A hard sub-15 MiB RSS target is not reachable in CPython: the bare interpreter is
already 11.8 MiB, and `ssl`, `http.client` and the full import set take it past
20 MiB before any work happens. What this project controls is the peak.

The measurements below were produced by `est doctor --memory`, which probes each
tier in a child interpreter and reads `/proc/self/statm`. They are reproducible
on the operator's own hardware — the table is not a claim from someone else's
laptop.

| Stage | RSS |
| --- | --- |
| bare interpreter | 11.8 MiB |
| + `ssl` | 13.9 MiB |
| + `http.client` | 15.4 MiB |
| + `sqlite3`, `tomllib`, `logging`, `dataclasses` | 20.5 MiB |
| + SQLite store open | 21.8 MiB |
| + engine + TLS context | 25.3 MiB |
| + one full poll (NOAA, GBIF, SAR, persist) | ~35.8 MiB |
| transient peak during a poll (VmHWM) | ~35.6 MiB |

Two findings from that table are worth stating plainly, because both were
initially wrong here and cost real time:

1. The first HTTPS request costs ~3.9 MiB **regardless of payload size**. It is
   the TLS context, not the data: the 1 KB F10.7 feed moves the needle exactly
   as much as a 2.5 MB RTSW file. Do not go looking for a big download to blame.
2. The NOAA peak is genuinely avoidable, and that is what the range prefetch is
   for. A naive `json.loads` of the full 2.5 MB feed peaks at ~27 MiB on its own.
   A `Range: bytes=0-32768` request with `Accept-Encoding: identity` cuts the
   transfer to 32 KiB and brings the whole NOAA source set to under 0.1 MiB above
   the import floor.

The default ceiling is 48 MiB (`daemon.max_rss_bytes`). A full poll measures
~35.8 MiB steady state against that budget, comfortably inside it. The RSS
warning threshold (`daemon.rss_warn_fraction = 0.85`) sits at ~41 MiB, above the
measured peak so the daemon does not warn on every pass, below the hard ceiling so
drift is caught before it becomes an OOM kill and restart loop.

```sh
uv run est doctor --memory   # per-stage breakdown, measured in child processes
```

## Development

```sh
uv run pytest                                     # unit + integration tests
uv run python -m pytest --doctest-modules src/emperor_space_tracker   # doctests
uv run ruff check src tests
uv run mypy
```

The suite covers the claims the README makes rather than the shape of the code:
`tests/test_catalogue.py` parses every shipped colony record and checks it is
complete and internally consistent, and `tests/test_integration.py` exercises a
real poll pass, the store write paths, latch persistence across a restart, the
cooldown and hysteresis gates, the daemon exit codes, and the Discord retry
ladder.

Ruff, strict mypy, the unit tests and the doctests are all expected to pass
before a change lands. The dashboard is covered by a smoke test that executes
the Streamlit script through `AppTest`, because an HTTP 200 from the Streamlit
server proves nothing: it serves its shell on every request and only runs the
script once a browser connects over a websocket.

## Known limits

- **Earth Engine is unverified end to end.** The synthetic backend is the tested
  path. Real GEE retrieval needs `earthengine authenticate` and has not been
  exercised against live data in this build.
- **AADC and SCAR expose no usable query API.** `colonies.toml` ships 13
  provenance-labelled records instead of polling a live census feed.
- **The Indian Ocean census is 41 years stale.** The last archipelago-wide
  survey of the Kerguelen gentoos was 1985; only the Courbet Peninsula has a
  recent count. Herman et al. 2020 name this sector one of the most critical
  data gaps for the gentoo complex, at ~14% of the global population. The
  Heard Island figure is from 1987 and its species assignment is inferred
  from tracking data, not genotypes — both are recorded in the record's `notes`.
- **`Pygoscelis kerguelensis` is a recent name.** Described in 2026 as
  superseding the 2020 treatment, which called the same population
  *P. taeniata*. Older sources citing *P. papua* for these colonies are a
  taxonomic difference, not a miscount.
- **No scatterer climatology.** Thresholds are physically-motivated constants in
  `sources/sar.py`, not derived from a Ross Sea sigma0 distribution. They are
  adequate for detecting *change* between consecutive passes, which is what the
  trend analysis does, but the absolute classification bands are not
  climatologically validated.
- **Sentinel-1 coverage is not uniform.** SAR works in darkness and through
  cloud, which is exactly why it is the right instrument for polar night, but the
  constellation's orbit tracks do not revisit every Ross Sea colony at a useful
  resolution on every pass. Repeat is roughly 12 hours where covered, and days
  where not. The most interesting freeze-up and melt transitions also happen in
  the sunlit months, so a polar-night series mostly shows stable consolidated
  ice with occasional lead events.

## Licence

MIT. See `LICENSE`.
