"""Tests for the HTTP transport: retry policy, caps, and the prefix scanner.

Every daemon pass goes through this module, and its two jobs are both about
failing in a bounded way. A node on a 5-minute poll in polar night cannot ask a
human what to do, so:

* **Bounded reads.** A 2.5 MB feed must not become a 2.5 MB buffer. The cap is
  checked *after* the read of ``cap + 1`` bytes, so an oversized body is
  detected without ever buffering all of it.
* **Bounded retries.** Rate limits and 5xx are retried with backoff; a 4xx is
  not, because a rejected request will be rejected again and retrying it burns
  the budget the next attempt needs.
* **Bounded scheme.** Only http and https are ever opened, checked once before
  any socket work rather than at each call site.

The retry tests replace ``time.sleep`` so the backoff schedule can be asserted
without the suite taking seconds of wall clock.
"""

from __future__ import annotations

import email.message
import gzip
import io
import json
import logging
import ssl
import time
import urllib.error
import urllib.request
import zlib
from typing import Any

import pytest

from emperor_space_tracker.errors import (
    AlertDeliveryError,
    SourceTimeoutError,
    SourceUnavailableError,
)
from emperor_space_tracker.net import (
    DEFAULT_PREFETCH_BYTES,
    HttpClient,
    HttpResponse,
    parse_json_array_prefix,
    require_http_url,
    retry_with_backoff,
    supports_range,
)
from emperor_space_tracker.sources.space_weather import SpaceWeatherClient

# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeHTTPResponse:
    """Minimal stand-in for what ``urlopen`` returns inside a ``with`` block."""

    def __init__(
        self,
        body: bytes = b"",
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        lines: list[bytes] | None = None,
    ) -> None:
        """Build a response with the given body, status and headers."""
        self._body = body
        self.status = status
        # email.message.Message rather than a dict: that is what urlopen
        # actually returns, and it is case-insensitively keyed. A plain dict
        # would make every header lookup in the client under test behave
        # differently from production while still looking correct.
        self.headers = email.message.Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value
        self._lines = lines

    def read(self, amount: int = -1) -> bytes:
        return self._body if amount < 0 else self._body[:amount]

    def geturl(self) -> str:
        return "https://example.invalid/final"

    def __enter__(self) -> FakeHTTPResponse:
        """Support use as a context manager, as ``urlopen`` results are."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close cleanly; nothing to release."""

    def __iter__(self) -> Any:
        """Iterate the scripted lines, as a streaming body would."""
        return iter(self._lines or [])


class FakeOpener:
    """Returns scripted responses or raises scripted errors, in order.

    Records every request so a test can assert *how many attempts* were made,
    which is the property the retry policy actually promises.
    """

    def __init__(self, script: list[Any]) -> None:
        """Take the scripted responses or errors, in order."""
        self.script = list(script)
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, timeout: float = 0.0) -> Any:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("the client made more attempts than the test scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(script: list[Any], **kwargs: Any) -> tuple[HttpClient, FakeOpener]:
    """Build a client with a scripted opener and no real backoff by default."""
    options: dict[str, Any] = {"max_retries": 3, "backoff_base": 0.0}
    options.update(kwargs)
    client = HttpClient(**options)
    opener = FakeOpener(script)
    client._opener = opener  # type: ignore[assignment]
    return client, opener


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make backoff instantaneous and observable.

    Without this the retry tests spend real seconds asleep, and a backoff that
    accidentally grew would slow the suite rather than fail it.
    """
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    _no_real_sleep.slept = slept  # type: ignore[attr-defined]


def _slept() -> list[float]:
    return getattr(_no_real_sleep, "slept", [])


def _ok(body: bytes, **kwargs: Any) -> FakeHTTPResponse:
    return FakeHTTPResponse(body, **kwargs)


def _http_error(code: int, reason: str = "Error") -> urllib.error.HTTPError:
    """Build the HTTPError that ``urlopen`` raises for a failure status."""
    return urllib.error.HTTPError(
        "https://example.invalid/x", code, reason, email.message.Message(), None
    )


def _install(client: HttpClient, result: Any) -> None:
    """Point ``client`` at an opener that yields ``result`` or raises it.

    Every transport test needs the same seam, and a per-test lambda both
    repeats the ignore comment and leaves mypy nothing to infer from.
    ``result`` may be a response, an exception to raise, or a callable
    producing a response so a test can inspect the request it was handed.
    """

    def opener(request: Any, **kwargs: Any) -> Any:
        if isinstance(result, Exception):
            raise result
        return result(request, **kwargs) if callable(result) else result

    client._opener.open = opener  # type: ignore[method-assign, assignment]


# --------------------------------------------------------------------------- #
# Scheme validation
# --------------------------------------------------------------------------- #


def test_only_http_and_https_are_ever_opened() -> None:
    """The single gate on what this client will fetch from.

    Validated in one place rather than at each call site, because a source that
    forgets is how ``file://`` or a data: URL reaches a socket opener.
    """
    for url in ("http://example.invalid/a", "https://example.invalid/a"):
        assert require_http_url(url) == url

    for url in ("file:///etc/shadow", "ftp://example.invalid/x", "gopher://x/1"):
        with pytest.raises(ValueError, match="non-HTTP URL scheme"):
            require_http_url(url)


@pytest.mark.parametrize("url", ["data:,plain", "mailto:a@b.c", "javascript:alert(1)"])
def test_a_scheme_without_a_double_slash_is_still_refused(url: str) -> None:
    """Partitioning on "://" would call these schemeless and admit them later.

    ``urlsplit`` is used precisely so these are recognised as carrying a scheme
    that is not in the allowed set.
    """
    with pytest.raises(ValueError, match="non-HTTP URL scheme"):
        require_http_url(url)


def test_a_relative_url_is_refused_as_not_absolute() -> None:
    with pytest.raises(ValueError, match="not absolute"):
        require_http_url("/json/f107_cm_flux.json")


def test_a_non_http_url_fails_before_any_retry_or_socket_work() -> None:
    """The check happens once, ahead of the loop, not per attempt."""
    client, opener = _client([])
    with pytest.raises(SourceUnavailableError, match="non-HTTP URL scheme"):
        client.get("file:///etc/shadow")
    assert opener.requests == []


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_a_retryable_status_is_retried_then_succeeds(code: int) -> None:
    """Rate limits and 5xx are transient; the next attempt usually works."""
    client, opener = _client([_http_error(code), _ok(b'{"ok": true}')])

    response = client.get("https://example.invalid/a")

    assert response.json() == {"ok": True}
    assert len(opener.requests) == 2
    assert len(_slept()) == 1


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_a_client_error_is_not_retried(code: int) -> None:
    """A rejected request will be rejected again.

    Retrying spends the node's budget on an outcome already known, and on a
    field connection that costs time no other product can use.
    """
    client, opener = _client([_http_error(code)])

    with pytest.raises(SourceUnavailableError, match=f"HTTP {code}"):
        client.get("https://example.invalid/a")

    assert len(opener.requests) == 1
    assert _slept() == []


def test_every_retryable_attempt_failing_raises_rather_than_looping_forever() -> None:
    client, opener = _client([_http_error(503), _http_error(503), _http_error(503)])

    with pytest.raises(SourceUnavailableError, match="HTTP 503"):
        client.get("https://example.invalid/a")

    assert len(opener.requests) == 3


def test_a_timeout_is_retried_and_then_reported_as_a_timeout() -> None:
    """A hung socket is worth another try; three failures are not."""
    client, opener = _client([TimeoutError(), TimeoutError(), TimeoutError()])

    with pytest.raises(SourceTimeoutError, match="timed out"):
        client.get("https://example.invalid/a")

    assert len(opener.requests) == 3


def test_a_wrapped_socket_timeout_is_reported_as_a_timeout_not_a_dead_host() -> None:
    """Unwrap the socket timeout that urllib reports inside a URLError.

    ``URLError`` with a ``socket.timeout`` reason is a timeout. Reporting it as
    "unreachable" would send an operator looking at DNS and routing instead of
    at the endpoint.
    """
    client, _ = _client([urllib.error.URLError(TimeoutError("timed out"))] * 3)

    with pytest.raises(SourceTimeoutError, match="timed out"):
        client.get("https://example.invalid/a")


def test_an_unreachable_host_is_retried_then_reported_as_unreachable() -> None:
    client, opener = _client(
        [urllib.error.URLError("Name or service not known")] * 3
    )

    with pytest.raises(SourceUnavailableError, match="unreachable"):
        client.get("https://example.invalid/a")

    assert len(opener.requests) == 3


def test_a_tls_failure_is_retried_then_reported() -> None:
    """A transient TLS failure on a captive portal or flaky link is retryable."""
    client, opener = _client([ssl.SSLError("handshake failure")] * 3)

    with pytest.raises(SourceUnavailableError, match="handshake"):
        client.get("https://example.invalid/a")

    assert len(opener.requests) == 3


def test_the_backoff_grows_exponentially_between_attempts() -> None:
    """Delay doubles per attempt, so a struggling endpoint is not hammered."""
    client, _ = _client(
        [_http_error(503), _http_error(503), _http_error(503)],
        backoff_base=0.5,
    )

    with pytest.raises(SourceUnavailableError):
        client.get("https://example.invalid/a")

    # 0.5 * 2**0, then 0.5 * 2**1. No sleep after the final attempt.
    assert _slept() == [0.5, 1.0]


def test_a_single_attempt_client_does_not_sleep_at_all() -> None:
    """``max_retries=1`` means no retry, so no backoff delay either."""
    client, opener = _client([_http_error(503)], max_retries=1)

    with pytest.raises(SourceUnavailableError):
        client.get("https://example.invalid/a")

    assert len(opener.requests) == 1
    assert _slept() == []


# --------------------------------------------------------------------------- #
# Response capture and caching
# --------------------------------------------------------------------------- #


def test_the_age_header_marks_a_cdn_cached_response() -> None:
    """A cached body is a successful request for a stale document.

    The node needs to know, because the staleness check downstream is what
    turns this flag into an operator-visible warning instead of a node that
    reports calm space weather while blind.
    """
    client, _ = _client([_ok(b"[]", headers={"Age": "3600"})])
    assert client.get("https://example.invalid/a").from_cache is True

    client, _ = _client([_ok(b"[]")])
    assert client.get("https://example.invalid/a").from_cache is False


def test_response_headers_are_lowercased_for_case_insensitive_lookup() -> None:
    """HTTP header names are case-insensitive; lookups should be too."""
    client, _ = _client([_ok(b"[]", headers={"X-Trace-Id": "abc123"})])
    response = client.get("https://example.invalid/a")
    assert response.headers["x-trace-id"] == "abc123"


def test_a_non_json_body_raises_a_source_error_naming_the_url() -> None:
    """An HTML error page from a CDN must not become a silent None."""
    client, _ = _client([_ok(b"<html>502</html>")])

    with pytest.raises(SourceUnavailableError, match="did not return valid JSON"):
        client.get_json("https://example.invalid/a")


# --------------------------------------------------------------------------- #
# The read cap
# --------------------------------------------------------------------------- #


def test_a_body_over_the_cap_is_refused_rather_than_buffered() -> None:
    """The cap is the memory guarantee the daemon's whole budget rests on.

    Reading ``cap + 1`` rather than the full body is deliberate: an endpoint
    that ignores the cap cannot make this client allocate more than the cap
    plus a single byte.
    """
    client, _ = _client([_ok(b"x" * 5000)], max_bytes=1000)

    with pytest.raises(SourceUnavailableError, match="exceeds the 1000 byte cap"):
        client.get("https://example.invalid/a")


def test_a_body_exactly_at_the_cap_is_accepted() -> None:
    """The comparison is strictly greater-than, so the boundary is usable."""
    client, _ = _client([_ok(b"x" * 1000)], max_bytes=1000)
    assert len(client.get("https://example.invalid/a")) == 1000


def test_a_per_call_cap_overrides_the_client_default() -> None:
    """`ping` needs a small cap on a huge feed; the default stays generous."""
    client, _ = _client([_ok(b"y" * 200)], max_bytes=100_000)

    # The per-call cap is 100, so the 200-byte body trips it even though the
    # client's own 100 KB default would have accepted it.
    with pytest.raises(SourceUnavailableError, match="exceeds the 100 byte cap"):
        client.get("https://example.invalid/a", max_bytes=100)


def test_a_gzip_body_is_transparently_decompressed() -> None:
    """The cap applies to the compressed read; the decoded form may be larger.

    That is fine and intended: the cap bounds what arrives on the wire, which
    is what the node's egress and memory budget are about.
    """
    payload = json.dumps([{"a": 1}])
    body = gzip.compress(payload.encode())
    client, _ = _client([_ok(body, headers={"Content-Encoding": "gzip"})], max_bytes=100_000)

    assert client.get_json("https://example.invalid/a") == [{"a": 1}]


def test_a_corrupt_gzip_body_is_reported_not_silently_truncated() -> None:
    client, _ = _client(
        [_ok(b"not gzip at all", headers={"Content-Encoding": "gzip"})], max_bytes=100_000
    )

    with pytest.raises(SourceUnavailableError, match=r"gzip body .* is corrupt"):
        client.get("https://example.invalid/a")


@pytest.mark.parametrize("wbits", [-zlib.MAX_WBITS, zlib.MAX_WBITS])
def test_deflate_is_decoded_with_or_without_the_zlib_wrapper(wbits: int) -> None:
    """Some origins send raw deflate, some wrap it; both must decode."""
    body = zlib.compress(b'[{"a": 1}]')
    if wbits == zlib.MAX_WBITS:
        body = zlib.compress(b'[{"a": 1}]')
    else:
        compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        body = compressor.compress(b'[{"a": 1}]') + compressor.flush()
    client, _ = _client([_ok(body, headers={"Content-Encoding": "deflate"})], max_bytes=100_000)

    assert client.get_json("https://example.invalid/a") == [{"a": 1}]


def test_a_corrupt_deflate_body_is_reported() -> None:
    client, _ = _client(
        [_ok(b"definitely not deflate", headers={"Content-Encoding": "deflate"})],
        max_bytes=100_000,
    )

    with pytest.raises(SourceUnavailableError, match=r"deflate body .* is corrupt"):
        client.get("https://example.invalid/a")


# --------------------------------------------------------------------------- #
# POST
# --------------------------------------------------------------------------- #


def test_a_post_sends_a_compact_json_body_and_returns_the_response() -> None:
    """Discord webhook delivery goes through here, so the body shape matters."""
    client, opener = _client([_ok(b'{"ok": true}', status=204)])
    seen: dict[str, Any] = {}

    def capture(request: urllib.request.Request, timeout: float = 0.0) -> Any:
        seen["data"] = request.data
        seen["content_type"] = request.headers.get("Content-type")
        return FakeHTTPResponse(b'{"ok": true}', status=204)

    client._opener.open = capture  # type: ignore[method-assign, assignment]
    response = client.post_json("https://example.invalid/hook", {"content": "storm"})

    assert response.status == 204
    assert json.loads(seen["data"]) == {"content": "storm"}
    assert b", " not in seen["data"], "separators should be compact"
    assert "json" in seen["content_type"]
    assert len(opener.requests) == 0


def test_a_post_to_a_non_http_url_is_refused_before_serialising() -> None:
    client, _ = _client([])
    with pytest.raises(AlertDeliveryError, match="non-HTTP URL scheme"):
        client.post_json("file:///etc/passwd", {"content": "x"})


def test_a_post_rejection_includes_the_upstream_detail() -> None:
    """A 400 from Discord says what it objected to, and the node should show it.

    The alert notifier retries nothing on a 4xx, so this message is the only
    clue an operator gets about why the page was not delivered.
    """
    error = urllib.error.HTTPError(
        "https://example.invalid/hook",
        400,
        "Bad Request",
        email.message.Message(),
        io.BytesIO(b'{"message":"payload is too long"}'),
    )
    client, _ = _client([error])

    with pytest.raises(SourceUnavailableError, match="payload is too long"):
        client.post_json("https://example.invalid/hook", {"content": "x"})


def test_a_post_transport_failure_names_the_failure() -> None:
    client, _ = _client([])
    _install(client, urllib.error.URLError("connection reset"))

    with pytest.raises(SourceUnavailableError, match="connection reset"):
        client.post_json("https://example.invalid/hook", {"content": "x"})


# --------------------------------------------------------------------------- #
# Line streaming
# --------------------------------------------------------------------------- #


def test_stream_lines_yields_decoded_lines_and_skips_blanks() -> None:
    client, _ = _client([])
    _install(client, FakeHTTPResponse(lines=[b'{"a": 1}\n', b"\n", b'{"a": 2}\n']))

    assert list(client.stream_lines("https://example.invalid/stream")) == [
        '{"a": 1}',
        '{"a": 2}',
    ]


def test_stream_lines_stops_at_its_limit_without_draining_the_source() -> None:
    """The point of streaming is bounded memory, so it must not read on."""
    client, _ = _client([])
    _install(client, FakeHTTPResponse(lines=[b"a\n", b"b\n", b"c\n", b"d\n"]))

    assert list(client.stream_lines("https://example.invalid/s", limit=2)) == ["a", "b"]


def test_undecodable_bytes_become_replacement_rather_than_raising() -> None:
    """A binary byte mid-stream must not kill a long-running reader."""
    client, _ = _client([])
    _install(client, FakeHTTPResponse(lines=[b'{"a": "ok"}\n', b"\xff\xfe\n"]))

    lines = list(client.stream_lines("https://example.invalid/s"))

    assert lines[0] == '{"a": "ok"}'
    assert "\ufffd" in lines[1]


def test_streaming_refuses_a_non_http_url() -> None:
    client, _ = _client([])
    with pytest.raises(SourceUnavailableError, match="non-HTTP URL scheme"):
        list(client.stream_lines("file:///etc/passwd"))


def test_a_stream_http_error_is_reported() -> None:
    client, _ = _client([])
    _install(client, _http_error(403, "Forbidden"))

    with pytest.raises(SourceUnavailableError, match="HTTP 403"):
        list(client.stream_lines("https://example.invalid/s"))


def test_a_stream_transport_failure_is_reported_mid_iteration() -> None:
    """A stream that dies after some lines still raises rather than ending quietly.

    A consumer that saw three lines and then silence would treat a dropped
    connection as end-of-file and believe the series was complete.
    """

    class DyingStream:
        def __enter__(self) -> DyingStream:
            """Support use as a context manager."""
            return self

        def __exit__(self, *exc: object) -> None:
            """Close cleanly; nothing to release."""

        def __iter__(self) -> Any:
            yield b"first\n"
            raise urllib.error.URLError("connection reset by peer")

    client, _ = _client([])
    _install(client, DyingStream())

    lines: list[str] = []
    with pytest.raises(SourceUnavailableError, match="connection reset"):
        for line in client.stream_lines("https://example.invalid/s"):
            lines.append(line)

    # The lines that did arrive are still delivered to the caller first.
    assert lines == ["first"]


# --------------------------------------------------------------------------- #
# ping
# --------------------------------------------------------------------------- #


def test_ping_uses_a_range_request_because_a_small_cap_would_look_like_an_outage() -> None:
    """`est doctor` must probe an endpoint the way the daemon reads it.

    NOAA's plasma feed is 2.5 MB. A capped GET would report the single most
    important upstream as down while the daemon itself is working fine, purely
    because the probe and the daemon disagree about how to read it.
    """
    client, _ = _client([])
    seen: dict[str, Any] = {}

    def capture(request: Any, **kwargs: Any) -> Any:
        seen["headers"] = request.headers
        return FakeHTTPResponse(b"x" * 1024, status=206)

    _install(client, capture)
    reachable, detail = client.ping("https://example.invalid/feed")

    assert reachable is True
    assert "206" in detail and "ranged" in detail
    assert seen["headers"]["Range"] == "bytes=0-1023"


def test_ping_reports_an_origin_that_ignored_the_range() -> None:
    """A 200 means the optimisation is inactive, which the operator should see."""
    client, _ = _client([FakeHTTPResponse(b"x" * 2048, status=200)])
    reachable, detail = client.ping("https://example.invalid/feed")

    assert reachable is True
    assert "range ignored" in detail


def test_ping_reports_a_cached_probe() -> None:
    client, _ = _client([FakeHTTPResponse(b"x", status=206, headers={"Age": "5"})])
    _, detail = client.ping("https://example.invalid/feed")
    assert "cached" in detail


def test_ping_never_raises_and_returns_the_reason_instead() -> None:
    """`est doctor` walks many endpoints; one failure must not abort the run."""
    client, _ = _client([_http_error(503)] * 3)

    reachable, detail = client.ping("https://example.invalid/feed")

    assert reachable is False
    assert "HTTP 503" in detail


# --------------------------------------------------------------------------- #
# The prefix scanner
# --------------------------------------------------------------------------- #


def test_a_truncated_trailing_record_is_dropped_not_guessed_at() -> None:
    """Half a record must not become a partial sample.

    This is the whole reason the scanner tracks brace depth instead of splitting
    on commas: a comma inside a value would otherwise split mid-record and
    produce a plausible-looking sample that is missing fields.
    """
    assert parse_json_array_prefix('[{"speed": 400}, {"speed": 4') == [{"speed": 400}]


def test_a_closing_bracket_inside_a_string_does_not_end_the_array() -> None:
    """String state must be tracked, or a value containing ']' truncates early."""
    raw = '[{"note": "ice] and more"}, {"note": "ok"}'
    assert parse_json_array_prefix(raw) == [{"note": "ice] and more"}, {"note": "ok"}]


def test_an_escaped_quote_does_not_end_a_string() -> None:
    raw = r'[{"note": "say \"hi\" now"}, {"note": "ok"}]'
    assert parse_json_array_prefix(raw) == [{"note": 'say "hi" now'}, {"note": "ok"}]


def test_nested_objects_and_arrays_are_counted_by_depth() -> None:
    raw = '[{"a": {"b": [1, 2]}}, {"c": 1}]'
    assert parse_json_array_prefix(raw) == [{"a": {"b": [1, 2]}}, {"c": 1}]


def test_a_complete_document_with_its_bracket_is_parsed_in_full() -> None:
    assert parse_json_array_prefix('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_a_document_with_no_records_yields_an_empty_list() -> None:
    assert parse_json_array_prefix("[]") == []
    assert parse_json_array_prefix("[") == []
    assert parse_json_array_prefix("not json at all") == []
    assert parse_json_array_prefix("") == []


def test_one_undecodable_record_does_not_lose_the_records_around_it() -> None:
    """A shape this scanner does not model costs one record, not the series.

    Losing everything would turn an upstream format change into a silently
    empty pass, which is the failure mode the project exists to avoid.
    """
    raw = '[{"a": 1}, {bad}, {"b": 2}]'
    assert parse_json_array_prefix(raw) == [{"a": 1}, {"b": 2}]


def test_the_scanner_tolerates_leading_text_before_the_array() -> None:
    """CDNs and proxies prepend things; the array is located, not assumed."""
    assert parse_json_array_prefix('\n\n[{"a": 1}, {"b":') == [{"a": 1}]


# --------------------------------------------------------------------------- #
# supports_range
# --------------------------------------------------------------------------- #


@pytest.fixture
def patched_urlopen(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace ``urlopen`` for the module-level ``supports_range`` probe.

    ``supports_range`` builds its own opener rather than going through an
    ``HttpClient``, so this is the seam it can be driven through. Returns a
    setter so each test can install its own response or exception.

    This was hand-rolled with try/finally and a saved reference before it was
    made a fixture. ``monkeypatch`` restores the original whatever happens,
    including on an assertion failure, which the manual version also did but by
    being duplicated four times.
    """

    def install(result: Any) -> None:
        def fake_urlopen(request: Any, **kwargs: Any) -> Any:
            if isinstance(result, Exception):
                raise result
            return result(request, **kwargs) if callable(result) else result

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    return install


def test_an_accept_ranges_bytes_header_means_ranges_are_available(
    patched_urlopen: Any,
) -> None:
    """The probe is a HEAD, so it never transfers the body it is asking about."""
    seen: dict[str, Any] = {}

    def capture(request: Any, **kwargs: Any) -> Any:
        seen["method"] = request.get_method()
        return FakeHTTPResponse(headers={"Accept-Ranges": "bytes"})

    patched_urlopen(capture)

    assert supports_range("https://example.invalid/a") is True
    assert seen["method"] == "HEAD"


def test_the_header_is_matched_case_insensitively(patched_urlopen: Any) -> None:
    """HTTP header names are case-insensitive; so is this comparison."""
    patched_urlopen(FakeHTTPResponse(headers={"accept-ranges": "BYTES"}))
    assert supports_range("https://example.invalid/a") is True


def test_a_206_response_implies_ranges_even_without_the_header(
    patched_urlopen: Any,
) -> None:
    """Some origins answer 206 to a HEAD without advertising Accept-Ranges."""
    patched_urlopen(FakeHTTPResponse(status=206))
    assert supports_range("https://example.invalid/a") is True


def test_an_origin_without_range_support_reports_false_rather_than_raising(
    patched_urlopen: Any,
) -> None:
    """A plain 200 with no Accept-Ranges means the optimisation is unavailable."""
    patched_urlopen(FakeHTTPResponse(status=200))
    assert supports_range("https://example.invalid/a") is False


def test_an_unreachable_origin_reports_false_rather_than_raising(
    patched_urlopen: Any,
) -> None:
    """`est doctor` must be able to say 'unknown', not crash on it."""
    patched_urlopen(urllib.error.URLError("unreachable"))
    assert supports_range("https://example.invalid/a") is False


def test_a_rejected_scheme_is_false_without_touching_the_network() -> None:
    assert supports_range("file:///etc/shadow") is False


# --------------------------------------------------------------------------- #
# The functional retry helper
# --------------------------------------------------------------------------- #


def test_retry_with_backoff_returns_the_first_success() -> None:
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "ok"

    assert retry_with_backoff(operation, attempts=3, backoff_base=0.0) == "ok"
    assert len(calls) == 3


def test_retry_with_backoff_reraises_the_last_failure() -> None:
    def operation() -> str:
        raise ValueError("still broken")

    with pytest.raises(ValueError, match="still broken"):
        retry_with_backoff(operation, attempts=2, backoff_base=0.0)


def test_retry_with_backoff_reports_each_retry_to_the_callback() -> None:
    """The callback is how a caller logs or pages without owning the loop."""
    seen: list[tuple[int, str]] = []

    def operation() -> str:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        retry_with_backoff(
            operation,
            attempts=3,
            backoff_base=0.0,
            on_retry=lambda attempt, exc: seen.append((attempt, str(exc))),
        )

    assert seen == [(1, "nope"), (2, "nope")]


def test_retry_with_backoff_does_not_sleep_after_the_final_attempt() -> None:
    def operation() -> str:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        retry_with_backoff(operation, attempts=3, backoff_base=0.5)

    assert _slept() == [0.5, 1.0]


def test_retry_with_backoff_returns_immediately_on_first_success() -> None:
    assert retry_with_backoff(lambda: "fine", attempts=3, backoff_base=5.0) == "fine"
    assert _slept() == []


# --------------------------------------------------------------------------- #
# Defaults and configuration
# --------------------------------------------------------------------------- #


def test_the_prefetch_default_leaves_the_read_cap_room_to_spare() -> None:
    """A range wider than the cap would be pointless: the cap truncates first.

    ``fetch_json_array_prefix`` raises its cap to ``max(prefetch, 64 KiB)``
    precisely so the range is what bounds the transfer, so the two defaults
    have to agree rather than one silently truncating the other.
    """
    assert DEFAULT_PREFETCH_BYTES == 64 * 1024
    assert DEFAULT_PREFETCH_BYTES >= 64 * 1024


def test_the_swpc_client_prefers_a_narrower_prefetch_than_the_transport_default() -> None:
    """32 KiB holds far more one-minute records than any poll consumes.

    The narrow value is the whole point of the range request: the live plasma
    feed is 2.5 MB, and reading a 32 KiB prefix costs 1.3% of that.
    """
    client = SpaceWeatherClient(HttpClient(max_retries=1))
    assert client.prefetch_bytes == 32 * 1024
    assert client.prefetch_bytes < DEFAULT_PREFETCH_BYTES


def test_disabling_tls_verification_warns_loudly_at_construction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A node behind a captive portal needs the option, and must be told.

    The warning is the only thing standing between an operator and an unnoticed
    MITM on their monitoring link, so it is asserted rather than assumed.
    """
    with caplog.at_level(logging.WARNING, logger="emperor.net"):
        HttpClient(verify_tls=False)

    assert any("TLS verification disabled" in r.message for r in caplog.records)


def test_a_client_with_verification_disabled_does_not_check_hostnames() -> None:
    """The lab-proxy path still needs an SSLContext, just a permissive one."""
    context = HttpClient._context(verify_tls=False)
    assert context.check_hostname is False
    assert context.verify_mode is ssl.CERT_NONE


def test_the_retry_loop_always_raises_on_its_final_attempt() -> None:
    """Every error class ends in a raise, so the loop has no fall-through tail.

    ``_request`` carries an unreachable ``raise`` after the loop for the
    mypy checker, because the compiler cannot see that each handler either
    retries or raises. This pins that belief: if a new error type were handled
    with a bare ``continue``, this fails rather than leaving a caller to
    receive ``None`` and report a successful fetch.
    """
    for error in (
        _http_error(503),
        TimeoutError(),
        urllib.error.URLError("unreachable"),
        urllib.error.URLError(TimeoutError()),
        ssl.SSLError("handshake"),
        OSError("disk on fire"),
    ):
        client, opener = _client([error] * 3)
        with pytest.raises((SourceUnavailableError, SourceTimeoutError)):
            client.get("https://example.invalid/a")
        assert len(opener.requests) == 3


def test_a_verifying_client_keeps_hostname_and_certificate_checks() -> None:
    """The default must not be the one that disables them."""
    context = HttpClient._context(verify_tls=True)
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_a_response_reports_its_body_length() -> None:
    response = HttpResponse(
        status=200, url="https://x", body=b"12345", elapsed_ms=1, headers={}
    )
    assert len(response) == 5
    assert response.json() == 12345


def test_a_response_prefix_decode_tolerates_a_ragged_tail() -> None:
    response = HttpResponse(
        status=206, url="https://x", body=b'[{"a": 1}, {"b":', elapsed_ms=1, headers={}
    )
    assert response.json_array_prefix() == [{"a": 1}]


def test_a_response_prefix_decode_survives_undecodable_bytes() -> None:
    """A truncated multibyte character must not raise on the way past.

    The prefix arrives at an arbitrary byte offset, so it can cut a multi-byte
    UTF-8 sequence in half. The scan replaces rather than decodes strictly,
    because dropping a stray replacement character is strictly better than
    failing the whole fetch over one bad byte at the tail.
    """
    body = '[{"a": "ok"}, {"b": "caf\u00e9'.encode()[:-1] + b"\xff"
    response = HttpResponse(
        status=206, url="https://x", body=body, elapsed_ms=1, headers={}
    )

    assert isinstance(response.json_array_prefix(), list)
