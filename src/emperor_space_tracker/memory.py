"""Resident-set accounting and the arena-return policy for a long-lived daemon.

The design target for this project is a node that runs for months in polar night
on whatever hardware is available, unattended. Two things follow from that.

Measure, do not assume
----------------------
:func:`measure_breakdown` imports each dependency tier in a *child* interpreter
and reads ``/proc/self/statm``, producing the table that ``est doctor --memory``
prints. An operator provisioning a node should be able to confirm the footprint
on their own silicon rather than trusting a number from someone else's laptop.

Return what you can
-------------------
Between polls the daemon has just built and discarded a batch of JSON objects.
CPython's obmalloc holds the freed arenas rather than returning them, so a
process that peaked at 27 MiB stays at 22 MiB. :func:`collect` forces a
collection and, where available, calls ``malloc_trim(0)`` to hand free pages
back to the kernel.

Measured on CPython 3.13.15 / x86_64 for this project's import set:

===========================  ========
after                        RSS
===========================  ========
bare interpreter              11.8 MiB
steady state after a poll     21.0 MiB
peak, before ``collect``      27.2 MiB
after ``collect``             21.9 MiB
===========================  ========

``malloc_trim`` is attempted but honestly does little on top of the collector
(< 0.2 MiB); it is retained because on musl-based Alpine, where glibc's trim is
absent, the *collector* alone is what matters and the code path is identical.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "MemorySample",
    "collect",
    "current_rss_bytes",
    "measure_breakdown",
    "peak_rss_bytes",
    "rss_mib",
]

_LOG = logging.getLogger("emperor.memory")

#: Import tiers probed by :func:`measure_breakdown`, in the order a process
#: actually pays for them. Each tier is cumulative.
_TIERS: Final[tuple[tuple[str, str], ...]] = (
    ("bare interpreter", "pass"),
    ("+ ssl", "import ssl"),
    ("+ http.client", "import ssl, http.client"),
    (
        "+ full daemon import set",
        "import ssl, http.client, json, sqlite3, tomllib, logging.handlers, "
        "dataclasses, math, gzip, zlib, argparse, datetime, enum, signal, re",
    ),
)

_LIBC: Final = "libc.so.6"
_MUSL: Final = "libc.musl-x86_64.so.1"
_MUSL_AARCH64: Final = "libc.musl-aarch64.so.1"

# ctypes exposes no public base class for a bound foreign function, so
# this is Any by necessity rather than by convenience.
_trim_handle: Any = None
_trim_probed = False


def _page_size() -> int:
    """Return the OS page size in bytes, defaulting to 4096."""
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 4096


def current_rss_bytes() -> int | None:
    """Return current resident set size in bytes, or ``None`` if unavailable.

    Returns
    -------
    int | None
        RSS in bytes from ``/proc/self/statm`` field 2, or ``None`` on a
        platform without procfs.
    """
    try:
        with Path("/proc/self/statm").open(encoding="ascii") as handle:
            fields = handle.read().split()
        return int(fields[1]) * _page_size()
    except (OSError, ValueError, IndexError):
        return None


def peak_rss_bytes() -> int | None:
    """Return peak resident set size in bytes from ``/proc/self/status``.

    ``VmHWM`` is the kernel's high-water mark for the process. Note that this is
    a *peak*: unlike RSS it does not fall when memory is released, which is why
    it is the number systemd's ``MemoryMax`` ultimately has to respect.

    Returns
    -------
    int | None
        Peak RSS in bytes, or ``None`` if unavailable.
    """
    try:
        with Path("/proc/self/status").open(encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def rss_mib(num_bytes: int | None) -> str:
    """Format a byte count as MiB for display.

    Parameters
    ----------
    num_bytes
        The value to format, or ``None``.

    Returns
    -------
    str
        A right-aligned MiB string, or ``"n/a"``.

    Examples
    --------
    >>> rss_mib(32 * 1024 * 1024)
    '32.00'
    >>> rss_mib(None)
    'n/a'
    """
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / (1024 * 1024):.2f}"


def _load_trim() -> Any:
    """Resolve ``malloc_trim`` from libc, caching the result.

    Returns
    -------
    Any
        A ``ctypes`` function pointer, or ``None`` when the platform's libc does
        not export it (notably musl, which never has).
    """
    # deliberate module-level cache
    global _trim_handle, _trim_probed
    if _trim_probed:
        return _trim_handle
    _trim_probed = True
    for name in (_LIBC, _MUSL, _MUSL_AARCH64):
        try:
            libc = ctypes.CDLL(name, use_errno=True)
            handle = libc.malloc_trim
            handle.argtypes = [ctypes.c_size_t]
            handle.restype = ctypes.c_int
            _trim_handle = handle
        except (OSError, AttributeError):
            continue
        else:
            break
    if _trim_handle is None:
        _LOG.debug("malloc_trim unavailable on this libc; relying on gc.collect only")
    return _trim_handle


def collect(*, trim: bool = True) -> tuple[int | None, int | None]:
    """Release reclaimable memory back to the operating system.

    Called between polls, never during one. Collect the daemon's transient
    garbage (parsed JSON records, formatted alert payloads) and ask libc to
    return the freed arenas.

    Parameters
    ----------
    trim
        Attempt ``malloc_trim(0)``. Pass ``False`` to skip the trim when
        benchmarking allocator behaviour.

    Returns
    -------
    tuple[int | None, int | None]
        ``(rss_before, rss_after)`` in bytes; either may be ``None``.

    Examples
    --------
    >>> before, after = collect()  # doctest: +SKIP
    >>> after <= before  # doctest: +SKIP
    True
    """
    before = current_rss_bytes()
    gc.collect()
    if trim:
        handle = _load_trim()
        if handle is not None:
            try:
                handle(0)
            except OSError as exc:  # pragma: no cover - platform dependent
                _LOG.debug("malloc_trim failed: %s", exc)
    return before, current_rss_bytes()


@dataclass(frozen=True, slots=True)
class MemorySample:
    """One row of the import-cost breakdown.

    Attributes
    ----------
    label
        Human description of the tier.
    rss_bytes
        Measured RSS with that tier imported.
    """

    label: str
    rss_bytes: int | None

    @property
    def rss_mib(self) -> str:
        """Return the measurement as a MiB string."""
        return rss_mib(self.rss_bytes)

    @property
    def available(self) -> bool:
        """Return whether the measurement succeeded."""
        return self.rss_bytes is not None


def _probe(script: str) -> int | None:
    """Run ``script`` in a child interpreter and return its RSS in bytes.

    A child process is essential: once the parent has imported ``ssl`` there is
    no way to un-import it, so an in-process measurement could only ever report
    a monotonic ratchet rather than a per-tier cost.
    """
    try:
        # fixed argv, no shell
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                f"{script}\n"
                f"st=open('/proc/self/statm').read().split()\n"
                f"print(int(st[1])*{_page_size()})",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def measure_breakdown() -> list[MemorySample]:
    """Measure the RSS cost of each dependency tier in child interpreters.

    Returns
    -------
    list[MemorySample]
        One cumulative row per tier, in import order. The last row is the
        daemon's steady-state floor before it does any work.

    Examples
    --------
    >>> rows = measure_breakdown()  # doctest: +SKIP
    >>> rows[-1].label  # doctest: +SKIP
    '+ full daemon import set'
    """
    samples: list[MemorySample] = []
    for label, script in _TIERS:
        samples.append(MemorySample(label=label, rss_bytes=_probe(script)))
    return samples


def format_breakdown(samples: list[MemorySample]) -> str:
    """Render :func:`measure_breakdown` output as an aligned table.

    Parameters
    ----------
    samples
        Rows to render.

    Returns
    -------
    str
        A fixed-width table with a total column.
    """
    if not samples:
        return "no samples"
    width = max(len(s.label) for s in samples)
    header = f"  {'tier'.ljust(width)}  {'RSS':>9s}  {'delta':>9s}"
    rule = f"  {'-' * width}  {'-' * 9}  {'-' * 9}"
    lines = [header, rule]
    previous = 0.0
    for sample in samples:
        current = (sample.rss_bytes or 0) / (1024 * 1024)
        delta = current - previous
        previous = current
        marker = "" if sample.available else "  (probe failed)"
        lines.append(
            f"  {sample.label.ljust(width)}  {current:7.2f}MiB  {delta:+7.2f}MiB{marker}"
        )
    return "\n".join(lines)
