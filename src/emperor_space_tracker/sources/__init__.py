"""Data source adapters.

Each module in this package owns exactly one upstream API, exposes a
``fetch_*``/``poll`` method returning typed records from
:mod:`emperor_space_tracker.models`, and converts every foreseeable failure into
a :class:`~emperor_space_tracker.errors.SourceError` subclass. Nothing in this
package raises anything else, which is what lets the supervision loop in
:mod:`emperor_space_tracker.engine` treat a whole poll as recoverable.

No module here imports the dashboard stack, and none of them import a
third-party library at module scope. ``sar.py`` is the only one that can pull an
optional dependency, and it does so lazily inside the GEE backend.
"""

from __future__ import annotations

__all__ = [
    "biological",
    "sar",
    "space_weather",
]
