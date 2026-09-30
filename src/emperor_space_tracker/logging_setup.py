"""Structured logging for a headless, unattended service.

Three design points that matter on a field node:

* **stderr stays human-readable.** systemd journals it, so the daemon writes
  plain, timestamped, greppable lines there. Nothing but diagnostics.
* **The rotating file is optional and opt-in JSON.** A node that is scraped by
  something else gets machine-readable lines; a node that is not, keeps bytes
  off flash.
* **Secrets are redacted at the formatter, not the call site.** Every record
  passes through :class:`RedactingFormatter`, so a future contributor cannot
  accidentally log a webhook URL by formatting a config object.

Rotation is the stdlib :class:`logging.handlers.RotatingFileHandler` rather than
logrotate: a user unit has no root, no cron and no ``/etc/logrotate.d`` on
Alpine, so the process must be self-sufficient.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path
from typing import Any, Final

from .config import Config
from .errors import ConfigError

__all__ = [
    "get_rss_bytes",
    "setup_logging",
    "shutdown_logging",
]

_ROOT: Final = logging.getLogger("emperor")

#: Discord webhook URLs and anything else shaped like a bearer credential.
_SECRET_PATTERNS: Final = (
    re.compile(r"https://(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks/\S+"),
    re.compile(r"\b[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\b\s*[=:]\s*\S+"),
)

_REDACTED: Final = "[redacted]"


def _redact(text: str) -> str:
    """Replace credential-shaped substrings in ``text``."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


class RedactingFormatter(logging.Formatter):
    """Formatter that scrubs credentials from both the message and the args.

    Arguments are redacted after ``%``-interpolation rather than before, because
    a partially-interpolated record can splice a secret across the boundary
    between format string and argument.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format ``record`` with credential redaction applied."""
        try:
            rendered = super().format(record)
        except Exception:
            return f"<unformattable log record: {record.msg!r}>"
        return _redact(rendered)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, with redaction applied."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` to a single redacted JSON line."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": _redact(record.getMessage()),
        }
        if record.exc_info:
            payload["exc"] = _redact(self.formatException(record.exc_info))
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        try:
            return json.dumps(payload, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return json.dumps({"ts": payload["ts"], "level": "ERROR",
                               "msg": "log record was not JSON-serialisable"})


def get_rss_bytes() -> int | None:
    """Return this process's resident set size in bytes, or ``None``.

    Reads ``/proc/self/statm`` (field 2, resident pages) which is a single short
    read and needs no third-party dependency. Returns ``None`` on non-Linux
    platforms rather than guessing.

    Returns
    -------
    int | None
        Resident set size in bytes, or ``None`` if ``/proc`` is unavailable.
    """
    try:
        with Path("/proc/self/statm").open(encoding="ascii") as handle:
            fields = handle.read().split()
    except OSError:
        return None
    if len(fields) < 2:
        return None
    try:
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


class _RssFilter(logging.Filter):
    """Attach the current RSS to every record as a structured field.

    Costs one small ``/proc`` read per emitted line. The daemon's default
    verbosity produces a handful of lines per five-minute poll, so this is
    noise-cheap and it makes ``est status`` able to report a real high-water
    mark rather than a snapshot.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Stamp ``record`` with an ``rss_kib`` attribute."""
        rss = get_rss_bytes()
        record.rss_kib = rss // 1024 if rss is not None else -1
        return True


def setup_logging(config: Config, *, quiet: bool = False) -> logging.Logger:
    """Configure and return the package logger.

    Safe to call more than once: existing handlers on the package logger are
    removed first, so the CLI can reconfigure after parsing ``--verbose``.

    Parameters
    ----------
    config
        Supplies the level, rotating file target and JSON preference.
    quiet
        Suppress the stderr handler. Used by ``est poll --quiet`` when the
        caller wants only the machine-readable result.

    Returns
    -------
    logging.Logger
        The configured ``emperor`` logger.

    Examples
    --------
    >>> import logging
    >>> from emperor_space_tracker.config import load_config
    >>> logger = setup_logging(load_config(use_user_config=False))
    >>> logger.info("node online")
    >>> sorted(h.formatter.__class__.__name__ for h in logger.handlers)
    ['RedactingFormatter', 'RedactingFormatter']
    """
    for handler in list(_ROOT.handlers):
        _ROOT.removeHandler(handler)
        handler.close()

    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    _ROOT.setLevel(level)
    _ROOT.propagate = False

    rss_filter = _RssFilter()

    if not quiet:
        stderr_formatter = RedactingFormatter(
            fmt="%(asctime)s %(levelname)-7s [%(name)s] rss=%(rss_kib)dKiB %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        stderr_handler = logging.StreamHandler(stream=sys.stderr)
        stderr_handler.setFormatter(stderr_formatter)
        stderr_handler.addFilter(rss_filter)
        _ROOT.addHandler(stderr_handler)

    try:
        # Called for its side effect: the rotating log file's parent directory
        # may not exist yet on a freshly imaged node.
        config.paths.ensure()
        file_handler = logging.handlers.RotatingFileHandler(
            config.paths.log_path,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding="utf-8",
        )
        if config.logging.json_file:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                RedactingFormatter(
                    fmt="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S%z",
                )
            )
        file_handler.addFilter(rss_filter)
        _ROOT.addHandler(file_handler)
    except (OSError, ConfigError) as exc:
        # A read-only or full filesystem must not stop the node from running.
        # The supervisor's exit status, not a log file, is the contract.
        fallback = RedactingFormatter("%(asctime)s %(levelname)-7s %(message)s")
        warn = logging.StreamHandler(stream=sys.stderr)
        warn.setFormatter(fallback)
        warn.addFilter(rss_filter)
        _ROOT.addHandler(warn)
        _ROOT.warning("file logging disabled: %s", exc)

    return _ROOT


def shutdown_logging() -> None:
    """Flush and detach all handlers, releasing the open log file."""
    for handler in list(_ROOT.handlers):
        _ROOT.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except (OSError, ValueError):
            pass
