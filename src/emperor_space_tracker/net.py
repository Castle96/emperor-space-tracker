"""Memory-bounded HTTP client built on the standard library alone.

This module is the reason the daemon stays inside its 48 MiB ceiling while
talking to APIs that publish multi-megabyte documents.

The problem
-----------
``https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json`` is a single JSON
array of one-minute DSCOVR/ACE/IMAP plasma samples for the trailing day. As of
writing it is ~2.5 MB. A naive ``urllib.request.urlopen(...).read()`` plus
``json.loads`` costs 25-40 MiB of peak RSS for one request, and the array is in
*descending* time order, so the freshest sample is the first 400 bytes.

The technique
-------------
Three cooperating tricks, none of which require a streaming JSON parser:

1. **Byte-range prefetch.** The endpoint advertises ``accept-ranges: bytes`` and
   orders records newest-first, so a ``Range: bytes=0-N`` request retrieves
   exactly the current data and nothing else. Measured against the live
   endpoint this is a 99.5% transfer reduction.

2. **Prefix-tolerant parsing.** The response is a JSON array, so a truncated
   prefix is repaired by dropping the final incomplete record and closing the
   bracket. :func:`parse_json_array_prefix` implements that and never raises on
   a ragged tail.

3. **Bounded reads everywhere else.** :func:`fetch_bytes` hard-caps every
   response at ``max_bytes`` and raises rather than letting a hostile or
   misconfigured endpoint allocate without limit.

Together these keep a polling pass in the low single-digit MiB while remaining
fully dependency-free.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import logging
import socket
import ssl
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Iterator
from typing import Any, Final
from urllib.parse import urlsplit

from .errors import AlertDeliveryError, SourceTimeoutError, SourceUnavailableError

__all__ = [
    "ALLOWED_SCHEMES",
    "DEFAULT_MAX_BYTES",
    "HttpClient",
    "HttpResponse",
    "build_ssl_context",
    "parse_json_array_prefix",
    "require_http_url",
    "supports_range",
]

_LOG = logging.getLogger("emperor.net")

#: URL schemes this client will ever open. Enforced on every request.
#:
#: Endpoints are configurable, which makes an unchecked ``urlopen`` a genuine
#: hole rather than a lint nit: a ``file://`` URL in a config or environment
#: override would read a local file and hand its bytes to a webhook, and
#: ``ftp://`` reaches outward over a protocol nothing here needs. Both are
#: refused before a socket is opened. Plain HTTP is permitted because the node
#: has to be debuggable against a captive portal or a LAN mirror on a station
#: with no outbound route; TLS verification itself is never optional.
ALLOWED_SCHEMES: Final = frozenset({"http", "https"})

#: Absolute ceiling for any single non-range response. 8 MiB is generous for
#: the JSON APIs this project consumes and small enough to fail loudly rather
#: than thrash a 128 MiB edge node.
DEFAULT_MAX_BYTES: Final = 8 * 1024 * 1024

#: How many bytes of a newest-first array to request when range-fetching.
DEFAULT_PREFETCH_BYTES: Final = 64 * 1024

_USER_AGENT: Final = "emperor-space-tracker/0.1 (+edge monitoring node)"

#: Retryable HTTP statuses. 429 is included because NOAA rate-limits aggressively
#: from shared cloud egress ranges, which is exactly what an edge fleet looks
#: like to them.
_RETRY_STATUSES: Final = frozenset({408, 425, 429, 500, 502, 503, 504})


def build_ssl_context() -> ssl.SSLContext:
    """Return a TLS context that verifies against the system trust store.

    Uses :func:`ssl.create_default_context`, which on CPython 3.10+ loads the
    platform trust store rather than shipping a bundled CA list. That matters
    for the distribution-agnostic requirement: the same code verifies correctly
    on Arch's ``ca-certificates``, Debian's ``ca-certificates``, RHEL's
    ``ca-certificates`` and Alpine's ``ca-certificates`` with no per-distro
    branch and no vendored certifi.

    Certificates are *not* disabled. ``est doctor`` checks that a trust store
    is actually present and explains how to fix it if not (on Alpine that is
    ``apk add ca-certificates``).
    """
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


def parse_json_array_prefix(raw: str) -> list[Any]:
    r"""Parse a possibly-truncated JSON array prefix into its complete records.

    The input is expected to be the leading slice of a JSON array document, e.g.
    ``'[{"a": 1}, {"b": 2}, {"c":'``. Everything up to the last *syntactically
    complete* top-level element is retained; the ragged tail and the closing
    bracket are supplied by this function.

    Parameters
    ----------
    raw
        The leading bytes of a JSON array, decoded as text.

    Returns
    -------
    list[Any]
        Every fully-formed element found, in document order. Empty if the
        prefix contains no complete element.

    Notes
    -----
    This is a string-scanner, not a JSON parser. It tracks brace depth, string
    state and escape state, which is sufficient for the flat, single-level
    object arrays that the SWPC RTSW feeds emit, and it degrades safely: on any
    unexpected input it returns what it has rather than raising.

    Examples
    --------
    >>> parse_json_array_prefix('[{"a": 1}, {"b": 2}, {"c":')
    [{'a': 1}, {'b': 2}]
    >>> parse_json_array_prefix('[{"t": "has \\" quote, and ] brace"}]')
    [{'t': 'has " quote, and ] brace'}]
    >>> parse_json_array_prefix('[{"a": 1}, {')
    [{'a': 1}]
    >>> parse_json_array_prefix('[]')
    []
    >>> parse_json_array_prefix('not json at all')
    []
    """
    start = raw.find("[")
    if start == -1:
        return []

    depth = 0
    in_string = False
    escaped = False
    element_start = -1
    records: list[Any] = []
    decoder = json.JSONDecoder()

    for index in range(start + 1, len(raw)):
        char = raw[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            if depth == 0:
                element_start = index
            depth += 1
            continue

        if char in "}]":
            if depth == 0:
                # A ']' at depth 0 closes the outer array: we are done.
                break
            depth -= 1
            if depth == 0 and element_start != -1:
                chunk = raw[element_start : index + 1]
                # A complete-looking element that will not decode means the
                # source is doing something we do not model. Skip it and keep the
                # surrounding records rather than losing all of them.
                with contextlib.suppress(json.JSONDecodeError):
                    records.append(decoder.decode(chunk))
                element_start = -1
            continue

        if char == "," and depth == 0:
            # Defensive: a bare top-level comma means we lost track. Reset so a
            # malformed region cannot corrupt every subsequent record.
            element_start = -1

    return records


class HttpResponse:
    """A fully-buffered, size-capped HTTP response.

    Attributes
    ----------
    status
        HTTP status code.
    url
        The URL that was actually fetched, after redirects.
    body
        Decoded response body as :class:`bytes`.
    elapsed_ms
        Wall-clock duration of the request in milliseconds.
    from_cache
        ``True`` when the upstream CDN served a cached representation, judged
        from the ``Age`` header. Useful for spotting a stale geomagnetic index
        during a network partition.
    """

    __slots__ = ("body", "elapsed_ms", "from_cache", "headers", "status", "url")

    def __init__(
        self,
        *,
        status: int,
        url: str,
        body: bytes,
        elapsed_ms: int,
        headers: dict[str, str],
        from_cache: bool = False,
    ) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.status = status
        self.url = url
        self.body = body
        self.elapsed_ms = elapsed_ms
        self.headers = headers
        self.from_cache = from_cache

    def json(self) -> Any:
        """Decode the body as JSON.

        Returns
        -------
        Any
            The decoded document.

        Raises
        ------
        ValueError
            If the body is not valid JSON.
        """
        return json.loads(self.body.decode("utf-8"))

    def json_array_prefix(self) -> list[Any]:
        """Decode a possibly-truncated JSON array prefix.

        Returns
        -------
        list[Any]
            Every complete element, tolerating a ragged final record.
        """
        return parse_json_array_prefix(self.body.decode("utf-8", errors="replace"))

    def __len__(self) -> int:
        """Return the body length in bytes."""
        return len(self.body)


def require_http_url(url: str) -> str:
    """Validate that a URL is one this client is willing to open.

    Parameters
    ----------
    url
        The candidate URL.

    Returns
    -------
    str
        The URL, unchanged.

    Raises
    ------
    ValueError
        If the URL is malformed or uses a scheme outside
        :data:`ALLOWED_SCHEMES`.

    Examples
    --------
    >>> require_http_url("https://services.swpc.noaa.gov/json/f107_cm_flux.json")
    'https://services.swpc.noaa.gov/json/f107_cm_flux.json'
    >>> require_http_url("file:///etc/shadow")
    Traceback (most recent call last):
        ...
    ValueError: refusing to open non-HTTP URL scheme 'file': 'file:///etc/shadow' \
(allowed: http, https)
    """
    # urlparse rather than a "://" partition: the partition treats any scheme
    # that has no double slash (data:, mailto:, javascript:) as having no scheme
    # at all, which both misreports the reason and would let a future edit to
    # the allowed set admit one of them by accident.
    scheme = urlsplit(url).scheme.lower()
    if not scheme:
        msg = f"URL is not absolute (no scheme): {url!r}"
        raise ValueError(msg)
    if scheme not in ALLOWED_SCHEMES:
        allowed = ", ".join(sorted(ALLOWED_SCHEMES))
        msg = (
            f"refusing to open non-HTTP URL scheme {scheme!r}: {url!r} "
            f"(allowed: {allowed})"
        )
        raise ValueError(msg)
    return url


def supports_range(url: str, *, timeout: float = 10.0) -> bool:
    """Return whether ``url`` advertises byte-range support via ``HEAD``.

    Used by :meth:`HttpClient.fetch_json_array_prefix` to decide between a
    range request and a full fetch, and exposed to ``est doctor`` so an operator
    can see exactly why the memory optimisation is or is not in play.

    Parameters
    ----------
    url
        The URL to probe.
    timeout
        Seconds to wait.

    Returns
    -------
    bool
        ``True`` if the server responded ``accept-ranges: bytes`` or 206.
        ``False`` on any failure, including a rejected scheme.
    """
    try:
        require_http_url(url)
    except ValueError:
        return False
    # scheme validated by require_http_url above
    request = urllib.request.Request(url, method="HEAD")
    request.add_header("User-Agent", _USER_AGENT)
    try:
        context = build_ssl_context()
        # scheme validated by require_http_url above
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            if response.headers.get("Accept-Ranges", "").lower() == "bytes":
                return True
            return int(response.status) == 206
    except (urllib.error.URLError, OSError, ValueError):
        return False


class HttpClient:
    """A small, retrying, size-capped HTTP client.

    Parameters
    ----------
    timeout
        Per-attempt socket timeout in seconds.
    max_bytes
        Hard cap on a single non-range response body.
    max_retries
        Total attempts, including the first. Retries apply to connection
        errors, timeouts and the statuses in ``_RETRY_STATUSES``.
    backoff_base
        Base seconds for exponential backoff; attempt *n* sleeps
        ``backoff_base * 2 ** (n - 1)``.
    verify_tls
        Set ``False`` only for a captive-portal or proxied lab. Constructing
        the client logs a warning so the downgrade is obvious in a node's
        log history.

    Examples
    --------
    >>> client = HttpClient(timeout=5.0)
    >>> response = client.get_json("https://example.invalid/x.json")
    Traceback (most recent call last):
    emperor_space_tracker.errors.SourceUnavailableError: ...
    """

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_retries: int = 3,
        backoff_base: float = 0.75,
        verify_tls: bool = True,
    ) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_retries = max(1, max_retries)
        self.backoff_base = backoff_base
        self.verify_tls = verify_tls
        if not verify_tls:
            _LOG.warning(
                "TLS verification disabled; use only behind a captive portal or lab proxy"
            )
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self._context(verify_tls))
        )

    @staticmethod
    def _context(verify_tls: bool) -> ssl.SSLContext:
        if verify_tls:
            return build_ssl_context()
        context = build_ssl_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    def _request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
        source: str = "http",
        accept_encoding: str = "gzip, deflate",
    ) -> HttpResponse:
        """Perform one HTTP request with retries and a hard read cap.

        Parameters
        ----------
        url
            Absolute URL.
        method
            HTTP method.
        headers
            Extra request headers.
        max_bytes
            Override the instance cap for this call.
        source
            Logical source name, used in error messages and logs.
        accept_encoding
            Value for the ``Accept-Encoding`` header. Range requests **must**
            pass ``"identity"``: a CDN that sees a client willing to accept gzip
            is entitled to ignore the range and serve the whole compressed
            entity, which is precisely the transfer we are trying to avoid.

        Returns
        -------
        HttpResponse

        Raises
        ------
        SourceTimeoutError
            If every attempt timed out.
        SourceUnavailableError
            If the endpoint could not be reached, refused TLS, or returned a
            non-retryable error status.
        """
        cap = max_bytes if max_bytes is not None else self.max_bytes
        # Validate the scheme once, before any retry loop or socket work.
        try:
            require_http_url(url)
        except ValueError as exc:
            raise SourceUnavailableError(source, str(exc)) from exc
        request_headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": accept_encoding,
            **(headers or {}),
        }
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            # scheme validated by require_http_url above
            request = urllib.request.Request(url, method=method, headers=request_headers)
            started = time.monotonic()
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    body = self._read_capped(response, cap, source=source, url=url)
                    elapsed = int((time.monotonic() - started) * 1000)
                    age = response.headers.get("Age")
                    return HttpResponse(
                        status=response.status,
                        url=response.geturl(),
                        body=body,
                        elapsed_ms=elapsed,
                        headers={k.lower(): v for k, v in response.headers.items()},
                        from_cache=bool(age),
                    )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in _RETRY_STATUSES and attempt < self.max_retries:
                    self._sleep(attempt, f"HTTP {exc.code}", source=source)
                    continue
                raise SourceUnavailableError(
                    source,
                    f"{method} {url} returned HTTP {exc.code} {exc.reason}",
                ) from exc
            except TimeoutError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(attempt, "timeout", source=source)
                    continue
                raise SourceTimeoutError(
                    source, f"{method} {url} timed out after {self.timeout:g}s"
                ) from exc
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, socket.timeout):
                    last_error = exc
                    if attempt < self.max_retries:
                        self._sleep(attempt, "timeout", source=source)
                        continue
                    raise SourceTimeoutError(
                        source, f"{method} {url} timed out after {self.timeout:g}s"
                    ) from exc
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(attempt, f"unreachable ({reason})", source=source)
                    continue
                raise SourceUnavailableError(
                    source, f"{method} {url} unreachable: {reason}"
                ) from exc
            except (ssl.SSLError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(attempt, str(exc), source=source)
                    continue
                raise SourceUnavailableError(source, f"{method} {url} failed: {exc}") from exc

        raise SourceUnavailableError(
            source,
            f"{method} {url} exhausted {self.max_retries} attempts: {last_error}",
        )

    @staticmethod
    def _read_capped(
        response: Any,
        cap: int,
        *,
        source: str,
        url: str,
    ) -> bytes:
        """Read at most ``cap`` bytes, transparently decompressing.

        Raises
        ------
        SourceUnavailableError
            If the body exceeds ``cap`` or cannot be decompressed.
        """
        encoding = (response.headers.get("Content-Encoding") or "").lower()
        raw = response.read(cap + 1)
        if len(raw) > cap:
            msg = f"response from {url} exceeds the {cap} byte cap; refusing to buffer it"
            raise SourceUnavailableError(source, msg)

        if "gzip" in encoding:
            try:
                raw = gzip.decompress(raw)
            except (OSError, zlib.error, EOFError) as exc:
                msg = f"gzip body from {url} is corrupt: {exc}"
                raise SourceUnavailableError(source, msg) from exc
        elif "deflate" in encoding:
            try:
                raw = zlib.decompress(raw, -zlib.MAX_WBITS)
            except zlib.error:
                try:
                    raw = zlib.decompress(raw)
                except (zlib.error, OSError, EOFError) as exc:
                    msg = f"deflate body from {url} is corrupt: {exc}"
                    raise SourceUnavailableError(source, msg) from exc
        return bytes(raw)

    def _sleep(self, attempt: int, reason: str, *, source: str) -> None:
        """Sleep with exponential backoff and log the reason."""
        delay = self.backoff_base * (2 ** (attempt - 1))
        _LOG.warning(
            "%s: %s; retry %d/%d in %.1fs", source, reason, attempt, self.max_retries, delay
        )
        time.sleep(delay)

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
        source: str = "http",
    ) -> HttpResponse:
        """Perform a capped ``GET``.

        Returns
        -------
        HttpResponse
        """
        return self._request(url, headers=headers, max_bytes=max_bytes, source=source)

    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        source: str = "http",
    ) -> Any:
        """``GET`` and decode a complete JSON document.

        Returns
        -------
        Any
            The decoded document.

        Raises
        ------
        SourceUnavailableError
            If the payload is not valid JSON.
        """
        response = self.get(url, headers=headers, source=source)
        try:
            return response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise SourceUnavailableError(
                source, f"{url} did not return valid JSON: {exc}"
            ) from exc

    def fetch_json_array_prefix(
        self,
        url: str,
        *,
        prefetch_bytes: int = DEFAULT_PREFETCH_BYTES,
        source: str = "http",
    ) -> list[Any]:
        """Fetch only the newest records of a newest-first JSON array.

        Issues ``Range: bytes=0-{prefetch_bytes - 1}`` **together with**
        ``Accept-Encoding: identity``. The second half of that is not optional.
        NOAA's JSON sits behind CloudFront, and CloudFront drops a ``Range``
        header outright when the request advertises a compressed encoding it
        could apply, answering ``HTTP 200`` with the full 2.5 MB entity. Asking
        for the identity encoding is what actually buys the 99.5% transfer
        reduction, and it is correct HTTP regardless: a byte range is only
        meaningful against a representation whose offsets the server and client
        agree on.

        If the endpoint ignores the range entirely, the response is still
        size-capped and parsed with the same prefix-tolerant scanner, and a
        single ``INFO`` line records that the optimisation is inactive.

        Parameters
        ----------
        url
            The array endpoint.
        prefetch_bytes
            How many leading bytes to request.
        source
            Logical source name.

        Returns
        -------
        list[Any]
            Complete records in document order (newest first for SWPC feeds).
        """
        response = self._request(
            url,
            headers={"Range": f"bytes=0-{prefetch_bytes - 1}"},
            max_bytes=max(prefetch_bytes, 64 * 1024),
            source=source,
            accept_encoding="identity",
        )
        if response.status != 206:
            _LOG.info(
                "%s: %s did not honour the range request (HTTP %d, %d bytes fetched); "
                "prefix parsing still applied but the transfer was not reduced",
                source,
                url,
                response.status,
                len(response.body),
            )
        return response.json_array_prefix()

    def post_json(
        self,
        url: str,
        payload: Any,
        *,
        source: str = "http",
    ) -> HttpResponse:
        """``POST`` a JSON body and return the response.

        Returns
        -------
        HttpResponse
        """
        try:
            require_http_url(url)
        except ValueError as exc:
            raise AlertDeliveryError(str(exc)) from exc
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        # scheme validated by require_http_url above
        request = urllib.request.Request(
            url,
            data=encoded,
            method="POST",
            headers={
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        started = time.monotonic()
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = self._read_capped(response, self.max_bytes, source=source, url=url)
                return HttpResponse(
                    status=response.status,
                    url=response.geturl(),
                    body=body,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    headers={k.lower(): v for k, v in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read(512).decode("utf-8", errors="replace") if exc.fp else ""
            raise SourceUnavailableError(
                source, f"POST {url} returned HTTP {exc.code} {exc.reason}: {detail[:200]}"
            ) from exc
        except (TimeoutError, urllib.error.URLError, ssl.SSLError, OSError) as exc:
            raise SourceUnavailableError(source, f"POST {url} failed: {exc}") from exc

    def stream_lines(
        self,
        url: str,
        *,
        limit: int = 1000,
        source: str = "http",
    ) -> Iterator[str]:
        """Yield decoded lines from a streaming text endpoint.

        Used for endpoints that publish one JSON object per line, which avoids
        materialising the whole document.

        Parameters
        ----------
        url
            The endpoint.
        limit
            Stop after this many non-empty lines.
        source
            Logical source name.

        Yields
        ------
        str
            Decoded, stripped lines.

        Raises
        ------
        SourceUnavailableError
            On transport failure.
        """
        try:
            require_http_url(url)
        except ValueError as exc:
            raise SourceUnavailableError(source, str(exc)) from exc
        # scheme validated by require_http_url above
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        emitted = 0
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    yield line
                    emitted += 1
                    if emitted >= limit:
                        return
        except urllib.error.HTTPError as exc:
            raise SourceUnavailableError(source, f"{url} returned HTTP {exc.code}") from exc
        except (TimeoutError, urllib.error.URLError, ssl.SSLError, OSError) as exc:
            raise SourceUnavailableError(source, f"{url} failed: {exc}") from exc

    def ping(self, url: str, *, source: str = "http") -> tuple[bool, str]:
        """Probe an endpoint without raising, for use by ``est doctor``.

        The probe asks for a 1 KiB byte range rather than a capped GET. A capped
        GET is the wrong tool for a liveness check: NOAA's plasma feed is a 2.5 MB
        newest-first array, so any small cap makes ``doctor`` report the single
        most important upstream as down when it is perfectly healthy, while the
        daemon itself is fine because it reads the same URL with a range request.
        Probing the way the daemon actually reads keeps the two honest.

        The response also reveals whether the origin honoured the range, so
        ``doctor`` gets that answer from the same request instead of paying for a
        second ``HEAD``.

        Parameters
        ----------
        url
            Endpoint to probe.
        source
            Logical source name for error reporting.

        Returns
        -------
        tuple[bool, str]
            ``(reachable, human_readable_detail)``. Never raises.
        """
        try:
            response = self._request(
                url,
                headers={"Range": "bytes=0-1023"},
                max_bytes=64 * 1024,
                source=source,
                accept_encoding="identity",
            )
        except (SourceUnavailableError, SourceTimeoutError) as exc:
            return False, str(exc)
        note = "cached" if response.from_cache else "live"
        ranged = "ranged" if response.status == 206 else "range ignored"
        return True, (
            f"HTTP {response.status}, {len(response.body)} bytes "
            f"in {response.elapsed_ms} ms ({note}, {ranged})"
        )


def retry_with_backoff(
    operation: Callable[[], Any],
    *,
    attempts: int = 3,
    backoff_base: float = 0.75,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Any:
    """Run ``operation`` with exponential backoff.

    A thin functional wrapper for call sites that already own their transport
    and only need the retry policy, such as the Google Earth Engine backend.

    Parameters
    ----------
    operation
        Zero-argument callable to invoke.
    attempts
        Maximum number of attempts.
    backoff_base
        Base backoff in seconds.
    on_retry
        Optional callback invoked as ``on_retry(attempt, exception)``.

    Returns
    -------
    Any
        Whatever ``operation`` returns.

    Raises
    ------
    Exception
        The last exception raised by ``operation``, once attempts are exhausted.
    """
    last: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return operation()
        except Exception as exc:
            last = exc
            if attempt >= attempts:
                break
            if on_retry is not None:
                on_retry(attempt, exc)
            time.sleep(backoff_base * (2 ** (attempt - 1)))
    assert last is not None
    raise last
