# The synthetic SAR simulator

`est seed` and the default `backend = "synthetic"` produce backscatter without
cloud credentials, so the pipeline, the alert rules and the dashboard are all
exercisable on a laptop. This note explains what the simulator actually models,
so a reader can tell which of its numbers mean something and which are invented
texture.

**Everything the simulator emits is stamped `provenance =
"synthetic.sar-simulator/v1"`.** It is never mixed with real observations, and
the dashboard shows a warning when the selected colony is backed by it. Nothing
here is a substitute for a Sentinel-1 acquisition.

## Why a simulator at all

The interesting failure modes in this project are all downstream of the radar:
the alert engine's hysteresis, the latch surviving a restart, the retention
policy, the dashboard's trend rendering. None of them can be tested without
backscatter arriving on a schedule. Requiring `earthengine authenticate` to
exercise them would mean those paths are only ever tested in production, which
is the failure mode this project exists to avoid.

## The field is four terms, layered

Each cell's sigma0 is a sum of four physical contributions, in the order a
modeller would build them. Each is seeded from a deterministic PRNG, so a given
(colony, date, seed) always regenerates byte-identical output.

**1. Consolidated-ice baseline.** Starts at −8.4 dB and brightens by up to
2.2 dB across the freezing season as brine drains from the upper ice and the
surface develops centimetre-scale roughness. Volume scattering in C band is
carried by brine channels and roughness elements, so both a thickening ice body
and a rougher surface raise sigma0.

**2. Orographic wind roughening.** A smooth directional gradient of up to
±2.6 dB, with a random orientation per acquisition. Antarctic katabatic flow is
directional and its cross-track signature dominates C-band sea-ice sigma0; the
orientation is redrawn per scene because real katabatic directions are not stable
between passes. This term is why two acquisitions of the same colony rarely
share a mean, and why the trend detector compares *consecutive* frames rather
than absolute values.

**3. Pressure ridges.** Sparse bright speckle (0.4% to 1.4% of cells per
acquisition, rising with wind) of +1.5 to +4.2 dB, where floes raft into linear
ridges. Ridges are the brightest surface in polar sea ice and they are
geometrically linear in reality; the simulator renders them as isolated pixels
because it has no directional structure at cell scale.

**4. Leads.** Dark channels grown from random seed points by a two-pass chamfer
distance transform, subtracting up to 9.5 dB. This is the term that carries the
project's actual signal. The distance field gives leads smooth nilas margins
rather than hard edges, and lead density rises from 3 to 25 seeds as the seasonal
breakup phase advances. A lead opening is what a *drop* in sigma0 between
acquisitions looks like from orbit, and it is the event the `fast-ice-retreat`
rule exists to catch.

A final Gaussian speckle term (σ ≈ 0.55 to 1.05 dB) keeps the field from
looking synthetic at cell level.

## The seasonal cycle is deliberately asymmetric

`seasonal_breakup_phase()` drives lead density and breakup. A cosine would
imply a smooth six-month melt, which is not how sea ice behaves:

| Period (austral) | Phase | State |
| --- | --- | --- |
| day 100–320 (Mar–Oct) | 0.00–0.06 | consolidated |
| day 320–380 (Nov–Dec) | 0.06–0.85 | accelerating breakup |
| day 380–400 (Dec–Jan) | 0.85–1.00 | near-total |
| day 40–100 (Feb–Mar) | 1.00–0.20 | ablating |

The long consolidation and short abrupt melt is the point. A colony that has to
sustain a ten-month breeding cycle on fast ice has its reproductive success
concentrated in the few weeks this function returns a high value, which is
exactly the exposure a monitoring node needs to see coming.

## Classification bands

`classify_cell()` partitions sigma0 into four states. These are
physically-motivated constants, not a derived climatology:

| Class | sigma0 | Meaning |
| --- | --- | --- |
| `open_water` | ≤ −11.5 dB | water or very thin nilas |
| `nilas` | ≤ −8.5 dB | new thin ice |
| `consolidated_ice` | ≤ −2.5 dB | the normal fast-ice state |
| `rough_ice` | above | ridged or wind-roughened |

**The honest limit:** these thresholds are not calibrated against a measured
Ross Sea sigma0 distribution. They are adequate for detecting *change* between
consecutive acquisitions, which is what the trend analysis does, and they are not
adequate for claiming an absolute ice state. A real deployment should fit them
to a local distribution before treating a classification as a finding.

## Revisit cadence, not poll cadence

The simulator acquires one frame per colony per `sar.revisit_hours` (12 h
default), matching Sentinel-1's repeat cycle over sea ice. It does *not*
acquire on every daemon poll.

That distinction is not a storage optimisation. The daemon polls every five
minutes because space weather genuinely moves on that cadence; the radar does
not, and no constellation produces a fresh Ross Sea frame because five minutes
elapsed. Tying acquisitions to the poll interval would manufacture 144 daily
observations from a sensor that flies twice, and a trend line built on invented
samples is worse than no trend line at all. See `Store.latest_acquisitions()`
for the gate and `test_a_colony_inside_its_revisit_window_is_not_re_imaged` for
the behaviour.

## What the simulator does not model

- **Speckle statistics.** Real single-look SAR speckle is multiplicative and
  Rayleigh-distributed; this is additive Gaussian. The simulator will not teach
  anything about speckle filtering.
- **Geometric correction.** No orbit-file correction, no radiometric terrain
  flattening, no incidence-angle dependence on sigma0. Real GRD products have
  these applied; this has none of them.
- **Polarisation.** `polarisation` is recorded on the scene but does not change
  the simulated field. Co-pol and cross-pol sigma0 differ substantially for
  sea ice.
- **Texture.** Pressure ridges are isolated bright pixels rather than linear
  features, and there is no floe-edge network. Any analysis relying on shape
  rather than intensity would find nothing here.

## Running it for real

```sh
uv sync --extra sar
earthengine authenticate
```

Then set `sar.backend = "gee"`. The GEE path filters `COPERNICUS/S1_GRD` (VV/VH)
or `COPERNICUS/S1_RAW` (HH/HV) by colony footprint, date and orbit pass, then
reduces each scene to a `grid_cells × grid_cells` sigma0 matrix with
`reduceRegion`.

**The GEE path is unverified end to end in this build.** It has been exercised
against the API's shape but not against a live authenticated retrieval. The
synthetic backend is the tested path.