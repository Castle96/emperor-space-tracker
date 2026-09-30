"""Alert evaluation, suppression and delivery.

The three parts are deliberately separated:

:class:`Alert`
    A value object describing something worth telling a human about.
:class:`~emperor_space_tracker.alerts.rules.RuleEngine`
    Decides *whether* to raise one, with hysteresis and cooldowns.
:class:`~emperor_space_tracker.alerts.notifier.DiscordNotifier`
    Decides *how* to say it.

The reason for that split is operational. During a storm the rules fire
repeatedly by design -- Kp stays at 6 for hours, and a naive threshold check
pages the duty operator every poll cycle. The suppression logic (hysteresis,
confirmation streaks, cooldowns, persistence across restarts) is the hard part
and it is testable without a network. The transport is replaceable.
"""

from __future__ import annotations

from .notifier import Alert, AlertDispatcher, DiscordNotifier, NullNotifier
from .rules import (
    RuleContext,
    RuleDefinition,
    RuleEngine,
    RuleOutcome,
    build_default_rules,
)

__all__ = [
    "Alert",
    "AlertDispatcher",
    "DiscordNotifier",
    "NullNotifier",
    "RuleContext",
    "RuleDefinition",
    "RuleEngine",
    "RuleOutcome",
    "build_default_rules",
]
