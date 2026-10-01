"""Tests for the NOAA SWPC client and its record parsers.

These paths carry almost no coverage, and that is not an ordinary gap. This is
the source that has been running live against the real service on every five
minute pass, and it is the one whose failure would page an operator. The
parsers in particular sit between a 2.5 MB third-party document and the alert
engine, so their job is to be exactly wrong in exactly the safe direction:

* a missing or sentinel value becomes ``None``, never a plausible number, so
  the alert rules see absence rather than ``9999`` km/s;
* a record from an inactive spacecraft is refused, so the node does not alert
  off a spacecraft SWPC has stopped publishing;
* one broken product degrades the snapshot instead of failing the pass.

Nothing here touches the network. The HTTP client is replaced with a stub that
returns fixture payloads, so these are tests about parsing and degradation
rather than about NOAA's current availability.
"""

from __future__ import annotations

import json
from datetime import UTC, timedelta
from typing import Any

import pytest

from emperor_space_tracker.errors import SourceError
from emperor_space_tracker.models import SpaceWeatherSnapshot, utc_now
from emperor_space_tracker.net import HttpClient, HttpResponse
from emperor_space_tracker.sources.space_weather import (
    SpaceWeatherClient,
    parse_f107,
    parse_kp,
    parse_magnetic_field,
    parse_plasma,
)

# --------------------------------------------------------------------------- #
# Fixtures: record shapes taken from the live RTSW / Kp / F10.7 documents
# --------------------------------------------------------------------------- #


def _plasma_record(**overrides: Any) -> dict[str, Any]:
    """Build a complete, active DSCOVR plasma record."""
    record = {
        "time_tag": "2026-01-01T12:00:00",
        "source": "DSCOVR",
        "active": True,
        "proton_speed": 412.5,
        "proton_density": 8.4,
        "proton_temperature": 14231.0,
        "overall_quality": 1,
    }
    record.update(overrides)
    return record


def _mag_record(**overrides: Any) -> dict[str, Any]:
    """Build a complete, active DSCOVR magnetic field record."""
    record = {
        "time_tag": "2026-01-01T12:00:00",
        "source": "DSCOVR",
        "active": True,
        "bt": 6.2,
        "bz_gsm": -3.1,
        "by_gsm": 4.4,
        "density_pct": 12.0,
    }
    record.update(overrides)
    return record


def _kp_record(**overrides: Any) -> dict[str, Any]:
    """Build a planetary Kp record with the nowcast present."""
    record = {
        "time_tag": "2026-01-01T12:00:00",
        "kp_index": 3,
        "estimated_kp": 3.33,
    }
    record.update(overrides)
    return record


class StubHttp:
    """Stands in for :class:`HttpClient` with fixture-driven payloads.

    Records every call so a test can assert *how* the client asked, not only
    what it returned. The prefix fetch in particular is only correct because of
    the two headers it sends, so "did it ask for the right thing" is a
    meaningful assertion rather than a mock formality.
    """

    def __init__(self, payloads: dict[str, Any]) -> None:
        """Record calls and answer from ``payloads`` keyed by URL suffix."""
        self.payloads = payloads
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def fetch_json_array_prefix(
        self, url: str, *, prefetch_bytes: int = 0, source: str = "http"
    ) -> list[Any]:
        self.calls.append(("prefix", url, {"prefetch_bytes": prefetch_bytes, "source": source}))
        value = self._lookup(url, source)
        if isinstance(value, Exception):
            raise value
        assert isinstance(value, list), "stubbed prefix payloads must be JSON arrays"
        return value

    def get_json(self, url: str, *, headers: Any = None, source: str = "http") -> Any:
        self.calls.append(("json", url, {"source": source}))
        value = self._lookup(url, source)
        if isinstance(value, Exception):
            raise value
        return value

    def _lookup(self, url: str, source: str) -> Any:
        for key, value in self.payloads.items():
            if url.endswith(key):
                return value
        raise AssertionError(f"unstubbed URL {url} for source {source}")

    def __getattr__(self, name: str) -> Any:
        """Refuse attribute access the real client does not have.

        Reached only if the source under test starts calling a transport method
        this stub does not model, which should fail loudly rather than resolve
        to a Mock and quietly return a truthy object the test then asserts on.
        """
        raise AttributeError(f"StubHttp does not implement {name!r}")

    def sources_called(self) -> list[str]:
        """Return the logical source name of each call, in order."""
        return [call[2]["source"] for call in self.calls]


# --------------------------------------------------------------------------- #
# The SWPC null convention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sentinel", [9999.0, 9999.9, -9999.0, 99999.0, 999.9])
def test_a_null_sentinel_never_becomes_a_measurement(sentinel: float) -> None:
    """SWPC's "value unavailable" must not read as a plausible wind speed.

    This is the single most dangerous thing the parser could get wrong. A
    sentinel that survived as ``9999.0`` would satisfy every storm threshold in
    the rules layer and page an operator at a fictitious super-fast solar wind.
    """
    sample = parse_plasma([_plasma_record(proton_speed=sentinel)])
    assert sample is not None
    assert sample.speed_kms is None


@pytest.mark.parametrize(
    "value",
    [None, True, False, "", "not-a-number", float("nan"), float("inf")],
)
def test_unusable_values_become_none_rather_than_raising(value: Any) -> None:
    """A malformed field degrades one sample, it does not fail the pass.

    ``bool`` is excluded from float coercion on purpose: ``True`` would become
    1.0 km/s, which is a real-looking measurement of nothing.
    """
    sample = parse_plasma([_plasma_record(proton_speed=value)])
    assert sample is not None
    assert sample.speed_kms is None


def test_a_partially_null_record_keeps_the_fields_that_are_real() -> None:
    """Absence is per-field, not per-sample.

    A missing density says nothing about the wind speed, and discarding the
    whole record would throw away a usable observation on every feed that
    intermittently omits one field.
    """
    sample = parse_plasma(
        [_plasma_record(proton_density=None, proton_temperature=9999.0, proton_speed=400.0)]
    )
    assert sample is not None
    assert sample.speed_kms == 400.0
    assert sample.density_per_cm3 is None
    assert sample.temperature_k is None


# --------------------------------------------------------------------------- #
# Active-source selection
# --------------------------------------------------------------------------- #


def test_an_inactive_spacecraft_is_never_the_measurement() -> None:
    """Only the spacecraft SWPC publishes as real-time may be read.

    The RTSW stream multiplexes DSCOVR, ACE and IMAP. An inactive record
    describes a spacecraft that has drifted or been decommissioned, and alerting
    off it would page someone about a measurement that is not being used by the
    operational model at all.
    """
    records = [
        _plasma_record(source="IMAP", active=False, proton_speed=8000.0),
        _plasma_record(source="ACE", active=False, proton_speed=7500.0),
    ]
    assert parse_plasma(records) is None
    assert parse_magnetic_field([r for r in records if "bt" in r]) is None


def test_the_active_record_wins_regardless_of_document_position() -> None:
    """Selection is by flag, not by first row.

    SWPC serves newest-first, but newest does not mean current: an active
    spacecraft can be followed by records from a source that has gone quiet.
    Reading the first row would take whichever happened to sort first.
    """
    records = [
        _plasma_record(source="IMAP", active=False, proton_speed=7000.0),
        _plasma_record(source="DSCOVR", active=True, proton_speed=350.0),
    ]
    sample = parse_plasma(records)
    assert sample is not None
    assert sample.source == "DSCOVR"
    assert sample.speed_kms == 350.0


def test_non_dict_rows_are_skipped_rather_than_crashing() -> None:
    """A ragged feed degrades to 'no usable record', not an exception."""
    assert parse_plasma(["", None, 42]) is None
    assert parse_magnetic_field(["", None, 42]) is None
    assert parse_kp(["", None, 42]) is None
    assert parse_f107(["", None, 42]) is None


def test_an_empty_document_yields_no_sample() -> None:
    assert parse_plasma([]) is None
    assert parse_magnetic_field([]) is None
    assert parse_kp([]) is None
    assert parse_f107([]) is None


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #


def test_a_naive_swpc_timestamp_is_read_as_utc_not_local_time() -> None:
    """SWPC publishes naive UTC; reading it as local shifts every observation.

    A node five hours from UTC that parses these as local timestamps silently
    reports its space weather five hours stale, which reads as a dead feed.
    """
    sample = parse_plasma([_plasma_record(time_tag="2026-01-01T12:00:00")])
    assert sample is not None
    assert sample.observed_at.utcoffset() == timedelta(0)
    assert sample.observed_at.hour == 12


def test_an_explicit_offset_denotes_the_instant_it_names() -> None:
    """An aware tag is honoured rather than reinterpreted as UTC.

    SWPC publishes naive UTC, so this path is theoretical for these feeds, but a
    tag that does carry an offset is unambiguous and must resolve to the instant
    it names. The property that matters is that it stays *aware*: a naive value
    here would be compared against aware datetimes downstream and raise.
    """
    sample = parse_plasma([_plasma_record(time_tag="2026-01-01T12:00:00+02:00")])
    assert sample is not None
    assert sample.observed_at.utcoffset() is not None
    # 12:00+02:00 is 10:00 UTC: the same moment, however it is spelled.
    assert sample.observed_at.astimezone(UTC).hour == 10


def test_a_z_suffix_is_accepted() -> None:
    sample = parse_plasma([_plasma_record(time_tag="2026-01-01T12:00:00Z")])
    assert sample is not None
    assert sample.observed_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("bad", [None, "", "not a date", 12345])
def test_an_unparseable_timestamp_does_not_drop_the_sample(bad: Any) -> None:
    """Losing the wind speed to fix a timestamp would be the wrong trade.

    The sample is stamped with the current time, which the staleness check then
    reports, and the operator sees "fresh but wrong time" rather than a node
    that has quietly stopped reporting wind at all.
    """
    sample = parse_plasma([_plasma_record(time_tag=bad)])
    assert sample is not None
    assert sample.speed_kms == 412.5
    assert sample.observed_at <= utc_now()


# --------------------------------------------------------------------------- #
# Kp
# --------------------------------------------------------------------------- #


def test_the_nowcast_is_preferred_over_the_settled_barometer_value() -> None:
    """The official 3-hour Kp is null while its own interval is still running.

    Falling back to the previous settled value would hold the node at storm
    level for up to three hours after a storm ends, or report calm conditions
    while one is building.
    """
    index = parse_kp([_kp_record(kp_index=None, estimated_kp=4.67)])
    assert index is not None
    assert index.value == 4.67
    assert index.is_estimated is True


def test_a_settled_barometer_value_is_used_when_no_nowcast_exists() -> None:
    index = parse_kp([_kp_record(kp_index=5, estimated_kp=None)])
    assert index is not None
    assert index.value == 5
    assert index.is_estimated is False


def test_the_newest_parseable_kp_record_wins() -> None:
    """SWPC publishes oldest-first, so the newest usable row is at the end."""
    records = [
        _kp_record(time_tag="2026-01-01T09:00:00", kp_index=1, estimated_kp=1.0),
        _kp_record(time_tag="2026-01-01T10:00:00", kp_index=2, estimated_kp=2.0),
        _kp_record(time_tag="2026-01-01T11:00:00", kp_index=3, estimated_kp=3.5),
    ]
    index = parse_kp(records)
    assert index is not None
    assert index.value == 3.5


def test_an_all_unusable_kp_document_yields_no_index() -> None:
    """Nothing parseable anywhere in the array means no index, not a guess.

    Falling back to a partial record here would put an unverified Kp into the
    store, and Kp drives the storm rule that pages an operator.
    """
    assert parse_kp([{"time_tag": "2026-01-01T12:00:00", "kp_index": None}]) is None


def test_a_null_only_kp_record_is_skipped_for_an_older_real_one() -> None:
    """The newest row can be all-null while its predecessor still has a value."""
    records = [
        _kp_record(time_tag="2026-01-01T10:00:00", kp_index=4, estimated_kp=4.0),
        _kp_record(time_tag="2026-01-01T11:00:00", kp_index=None, estimated_kp=None),
    ]
    index = parse_kp(records)
    assert index is not None
    assert index.value == 4.0


# --------------------------------------------------------------------------- #
# F10.7 radio flux
# --------------------------------------------------------------------------- #


def test_a_zero_or_null_flux_is_not_reported_as_the_newest_value() -> None:
    """Zero and the sentinel mean 'no measurement', not 'no solar flux'.

    Returning either would put a confident 0.0 sfu into the store during a
    period when the instrument simply had nothing to report.
    """
    assert parse_f107([{"time_tag": "2026-01-01", "flux": 0.0}]) is None
    assert parse_f107([{"time_tag": "2026-01-01", "flux": 9999.0}]) is None


def test_an_all_unusable_flux_document_yields_no_value() -> None:
    records = [
        {"time_tag": "2026-01-01T09:00:00", "flux": None},
        {"time_tag": "2026-01-01T11:00:00", "flux": 0.0},
    ]
    assert parse_f107(records) is None


def test_the_newest_positive_flux_is_returned() -> None:
    records = [
        {"time_tag": "2026-01-01T09:00:00", "flux": 84.0},
        {"time_tag": "2026-01-01T11:00:00", "flux": 91.5},
    ]
    assert parse_f107(records) == 91.5


# --------------------------------------------------------------------------- #
# Client behaviour against a stubbed transport
# --------------------------------------------------------------------------- #


def _client(payloads: dict[str, Any], **kwargs: Any) -> tuple[SpaceWeatherClient, StubHttp]:
    stub = StubHttp(payloads)
    return SpaceWeatherClient(stub, **kwargs), stub  # type: ignore[arg-type]


def test_a_healthy_poll_assembles_all_four_products() -> None:
    client, stub = _client({
        "rtsw_wind_1m.json": [_plasma_record()],
        "rtsw_mag_1m.json": [_mag_record()],
        "planetary_k_index_1m.json": [_kp_record()],
        "f107_cm_flux.json": [{"time_tag": "2026-01-01T12:00:00", "flux": 88.0}],
    })

    snapshot, health = client.poll()

    assert snapshot.plasma is not None and snapshot.plasma.speed_kms == 412.5
    assert snapshot.magnetic_field is not None and snapshot.magnetic_field.bz_gsm_nt == -3.1
    assert snapshot.kp is not None and snapshot.kp.value == 3.33
    assert snapshot.f107_sfu == 88.0
    assert snapshot.degraded == ()
    assert all(h.ok for h in health)
    assert stub.sources_called() == ["swpc.plasma", "swpc.mag", "swpc.kp", "swpc.f107"]


def test_one_broken_product_degrades_the_snapshot_without_losing_the_rest() -> None:
    """A single upstream failure must not blank the whole pass.

    This is the property that lets a node keep reporting wind speed while the
    geomagnetic index feed is down. Failing the pass instead would leave the
    operator with nothing rather than with three good products and one warning.
    """
    client, _ = _client({
        "rtsw_wind_1m.json": [_plasma_record()],
        "rtsw_mag_1m.json": SourceError("swpc.mag", "HTTP 503"),
        "planetary_k_index_1m.json": [_kp_record()],
        "f107_cm_flux.json": [{"time_tag": "2026-01-01T12:00:00", "flux": 88.0}],
    })

    snapshot, health = client.poll()

    assert snapshot.plasma is not None
    assert snapshot.kp is not None
    assert snapshot.magnetic_field is None
    assert "magnetic_field" in snapshot.degraded
    assert {h.source: h.ok for h in health}["swpc.mag"] is False


def test_every_product_failing_still_yields_a_snapshot_naming_all_of_them() -> None:
    """The snapshot exists even with nothing in it, and says what is missing.

    A pass that raised would leave no record at all, so the operator could not
    distinguish 'the node is broken' from 'every upstream is broken'.
    """
    client, _ = _client({
        "rtsw_wind_1m.json": SourceError("swpc.plasma", "unreachable"),
        "rtsw_mag_1m.json": SourceError("swpc.mag", "unreachable"),
        "planetary_k_index_1m.json": SourceError("swpc.kp", "unreachable"),
        "f107_cm_flux.json": SourceError("swpc.f107", "unreachable"),
    })

    snapshot, health = client.poll()

    assert snapshot.plasma is None
    assert set(snapshot.degraded) == {"plasma", "magnetic_field", "planetary_k_index", "f10_7_flux"}
    assert not any(h.ok for h in health)


def test_a_feed_with_no_active_spacecraft_is_reported_in_plain_language() -> None:
    """The detail string says why, because "0 records" would be a dead end."""
    client, _ = _client({
        "rtsw_wind_1m.json": [_plasma_record(active=False)],
        "rtsw_mag_1m.json": [_mag_record(active=False)],
        "planetary_k_index_1m.json": [_kp_record()],
        "f107_cm_flux.json": [{"time_tag": "2026-01-01T12:00:00", "flux": 88.0}],
    })

    snapshot, health = client.poll()

    detail = next(h for h in health if h.source == "swpc.plasma").detail
    assert "none from an active spacecraft" in detail
    assert snapshot.degraded == ("plasma", "magnetic_field")


def test_a_kp_array_of_only_unusable_rows_is_reported_as_unparseable() -> None:
    """The detail says the array had no usable row, not just that it failed."""
    client, _ = _client({
        "rtsw_wind_1m.json": [_plasma_record()],
        "rtsw_mag_1m.json": [_mag_record()],
        "planetary_k_index_1m.json": [{"time_tag": "2026-01-01T12:00:00", "kp_index": None}],
        "f107_cm_flux.json": [{"time_tag": "2026-01-01T12:00:00", "flux": 70.0}],
    })

    snapshot, health = client.poll()

    assert snapshot.kp is None
    detail = next(h for h in health if h.source == "swpc.kp").detail
    assert "no parseable Kp record" in detail


def test_a_json_object_where_an_array_was_expected_is_refused() -> None:
    """A CDN error page served as JSON must not be scanned as an array."""
    client, _ = _client({
        "rtsw_wind_1m.json": [_plasma_record()],
        "rtsw_mag_1m.json": [_mag_record()],
        "planetary_k_index_1m.json": {"error": "maintenance"},
        "f107_cm_flux.json": {"error": "maintenance"},
    })

    snapshot, health = client.poll()

    assert snapshot.kp is None
    assert snapshot.f107_sfu is None
    assert all(
        "expected a JSON array" in h.detail
        for h in health
        if h.source in {"swpc.kp", "swpc.f107"}
    )


def test_f107_can_be_switched_off_for_a_very_constrained_node() -> None:
    """The fourth product is optional and must not be fetched when disabled."""
    client, stub = _client(
        {
            "rtsw_wind_1m.json": [_plasma_record()],
            "rtsw_mag_1m.json": [_mag_record()],
            "planetary_k_index_1m.json": [_kp_record()],
        },
        track_f107=False,
    )

    snapshot, health = client.poll()

    assert snapshot.f107_sfu is None
    assert "swpc.f107" not in stub.sources_called()
    assert not any(h.source == "swpc.f107" for h in health)
    assert snapshot.degraded == ()


def _patched_client(
    response: HttpResponse, seen: dict[str, Any] | None = None
) -> HttpClient:
    """Return a real client whose ``_request`` yields ``response``.

    The client is patched rather than mocked wholesale so the code under test
    still runs against the genuine ``fetch_json_array_prefix``, which is where
    the header decision lives.
    """
    client = HttpClient(max_retries=1)

    def fake_request(url: str, **kwargs: Any) -> HttpResponse:
        if seen is not None:
            seen.update(kwargs)
            seen["url"] = url
        return response

    client._request = fake_request  # type: ignore[method-assign]
    return client


def test_the_prefix_fetch_requests_a_range_and_the_identity_encoding() -> None:
    """The transfer saving depends on both headers, so both are asserted.

    A CDN that sees a client willing to accept compression is entitled to ignore
    the byte range and serve the whole 2.5 MB entity, which is the exact cost
    the range exists to avoid.
    """
    seen: dict[str, Any] = {}
    client = _patched_client(
        HttpResponse(
            status=206,
            url="https://example.invalid/array.json",
            body=b'[{"a": 1}, {"b":',
            elapsed_ms=1,
            headers={},
        ),
        seen,
    )
    assert client.fetch_json_array_prefix(
        "https://example.invalid/array.json", prefetch_bytes=4096
    ) == [{"a": 1}]

    assert seen["headers"] == {"Range": "bytes=0-4095"}
    assert seen["accept_encoding"] == "identity"
    assert seen["max_bytes"] == 64 * 1024


def test_an_origin_that_ignores_the_range_still_parses_and_stays_capped() -> None:
    """A server answering 200 with the full body is degraded, not fatal.

    The memory cap and the prefix scanner still apply, so the node keeps working
    on an endpoint that has stopped honouring ranges; only the transfer
    reduction is lost, and the log line says so.
    """
    body = json.dumps([_plasma_record(), _plasma_record(proton_speed=1.0)]).encode()
    client = _patched_client(
        HttpResponse(
            status=200,
            url="https://example.invalid/array.json",
            body=body,
            elapsed_ms=1,
            headers={},
        )
    )
    records = client.fetch_json_array_prefix("https://example.invalid/array.json")
    assert len(records) == 2


def test_the_observation_window_spans_the_products_that_arrived() -> None:
    """A window over the three samples, so the dashboard can show coherence."""
    client, _ = _client({
        "rtsw_wind_1m.json": [_plasma_record(time_tag="2026-01-01T12:00:00")],
        "rtsw_mag_1m.json": [_mag_record(time_tag="2026-01-01T12:04:00")],
        "planetary_k_index_1m.json": [_kp_record(time_tag="2026-01-01T12:02:00")],
        "f107_cm_flux.json": [{"time_tag": "2026-01-01T11:00:00", "flux": 88.0}],
    })

    snapshot, _ = client.poll()

    assert snapshot.window is not None
    assert snapshot.window.start < snapshot.window.end


def test_no_samples_yet_means_no_window_rather_than_a_zero_width_one() -> None:
    client, _ = _client({
        "rtsw_wind_1m.json": SourceError("swpc.plasma", "down"),
        "rtsw_mag_1m.json": SourceError("swpc.mag", "down"),
        "planetary_k_index_1m.json": SourceError("swpc.kp", "down"),
        "f107_cm_flux.json": SourceError("swpc.f107", "down"),
    })

    snapshot, _ = client.poll()
    assert snapshot.window is None


# --------------------------------------------------------------------------- #
# Staleness
# --------------------------------------------------------------------------- #


def _snapshot_with_age(**ages: timedelta | None) -> SpaceWeatherSnapshot:
    """Build a snapshot whose samples are the given ages old."""
    from emperor_space_tracker.models import (
        InterplanetaryMagneticField,
        KpIndex,
        SolarWindPlasma,
    )

    now = utc_now()
    plasma_age = ages.get("plasma")
    field_age = ages.get("magnetic_field")
    kp_age = ages.get("kp")
    plasma = (
        None
        if plasma_age is None
        else SolarWindPlasma(now - plasma_age, "DSCOVR", 400.0, 5.0, 1000.0, True, 1)
    )
    field = (
        None
        if field_age is None
        else InterplanetaryMagneticField(
            now - field_age, "DSCOVR", 5.0, -1.0, 2.0, 5.0, True
        )
    )
    kp = None if kp_age is None else KpIndex(now - kp_age, 3, 3.3)
    return SpaceWeatherSnapshot(now, plasma, field, kp, None, None, ())


def test_a_fresh_snapshot_reports_nothing_stale() -> None:
    client = SpaceWeatherClient(StubHttp({}))  # type: ignore[arg-type]
    snapshot = _snapshot_with_age(
        plasma=timedelta(minutes=1), magnetic_field=timedelta(minutes=2), kp=timedelta(minutes=3)
    )
    assert client.is_stale(snapshot, 3600.0) == []


def test_a_cached_stale_sample_is_caught_even_though_the_fetch_succeeded() -> None:
    """A CDN happily serves an hours-old body with a 200.

    This is the failure mode the check exists for: without it the node reports
    calm space weather while blind, which is worse than reporting a fault,
    because the operator has been told everything is fine.
    """
    client = SpaceWeatherClient(StubHttp({}))  # type: ignore[arg-type]
    snapshot = _snapshot_with_age(
        plasma=timedelta(hours=6), magnetic_field=timedelta(minutes=1), kp=timedelta(minutes=1)
    )

    stale = client.is_stale(snapshot, 3600.0)

    assert len(stale) == 1
    assert "plasma" in stale[0]
    assert "min old" in stale[0]


def test_a_missing_product_is_not_reported_as_stale() -> None:
    """Absent and old are different problems with different fixes."""
    client = SpaceWeatherClient(StubHttp({}))  # type: ignore[arg-type]
    snapshot = _snapshot_with_age(plasma=None, magnetic_field=None, kp=timedelta(minutes=1))
    assert client.is_stale(snapshot, 3600.0) == []


# --------------------------------------------------------------------------- #
# Kp history
# --------------------------------------------------------------------------- #


def test_kp_history_returns_only_the_requested_window() -> None:
    """The 72 h default exists to draw a barometer plot, not to replay years."""
    now = utc_now()
    fresh = _kp_record(
        time_tag=(now - timedelta(hours=2)).isoformat(), kp_index=2, estimated_kp=2.0
    )
    old = _kp_record(
        time_tag=(now - timedelta(hours=200)).isoformat(), kp_index=5, estimated_kp=5.0
    )
    client, _ = _client({"planetary_k_index_1m.json": [old, fresh]})

    series = client.recent_kp_history(hours=72)

    assert [k.value for k in series] == [2.0]


def test_kp_history_degrades_to_empty_rather_than_raising() -> None:
    """The dashboard must still render with no history at all."""
    client, _ = _client({"planetary_k_index_1m.json": SourceError("swpc.kp.history", "down")})
    assert client.recent_kp_history() == []


def test_kp_history_ignores_a_non_array_payload() -> None:
    client, _ = _client({"planetary_k_index_1m.json": {"error": "nope"}})
    assert client.recent_kp_history() == []


def test_kp_history_skips_rows_that_are_not_objects() -> None:
    """A row the scanner cannot read is skipped, not fatal to the series."""
    now = utc_now()
    good = _kp_record(time_tag=(now - timedelta(hours=1)).isoformat(), estimated_kp=2.5)
    client, _ = _client({"planetary_k_index_1m.json": ["", None, 7, good]})

    assert [k.value for k in client.recent_kp_history()] == [2.5]
