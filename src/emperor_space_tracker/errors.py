"""Exception hierarchy for the tracker.

Every failure mode the daemon can survive is a subclass of :class:`TrackerError`.
The supervision loop in :mod:`emperor_space_tracker.engine` catches that base
class, degrades gracefully and keeps polling; anything else is a genuine bug and
is allowed to propagate to the crash loop.
"""

from __future__ import annotations

__all__ = [
    "AlertDeliveryError",
    "ConfigError",
    "SourceError",
    "SourceTimeoutError",
    "SourceUnavailableError",
    "StoreError",
    "TrackerError",
]


class TrackerError(Exception):
    """Base class for every recoverable error raised by this package."""


class ConfigError(TrackerError):
    """Configuration file is missing, unparseable or semantically invalid."""


class StoreError(TrackerError):
    """The SQLite state store could not be opened, migrated or written."""


class SourceError(TrackerError):
    """A data source returned an unusable payload."""

    def __init__(self, source: str, message: str) -> None:
        """Record the logical source name alongside the message.

        Parameters
        ----------
        source
            Logical source name, such as ``"swpc.plasma"``. Kept as an
            attribute so health reporting can attribute a failure without
            string-parsing the exception.
        message
            Human-readable explanation.
        """
        super().__init__(f"{source}: {message}")
        self.source = source
        self.message = message


class SourceTimeoutError(SourceError):
    """A data source did not respond within the configured budget."""


class SourceUnavailableError(SourceError):
    """A data source could not be reached at all (DNS, TLS, refused, 5xx)."""


class AlertDeliveryError(TrackerError):
    """An alert channel rejected or failed to accept a payload."""
