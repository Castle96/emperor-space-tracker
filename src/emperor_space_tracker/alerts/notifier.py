"""Alert delivery channels.

:class:`Alert` is a transport-agnostic value object. :class:`AlertDispatcher`
is the protocol a channel must satisfy, and two implementations ship:

:class:`NullNotifier`
    Discards everything and records the payload. This is the default when no
    webhook is configured, and it is what makes the rule engine testable without
    a network. Critically, a node with no configured channel still *logs* every
    alert at its computed severity, so nothing is silently lost.
:class:`DiscordNotifier`
    Posts an embed to a webhook URL held in an environment variable.

Two implementation details worth stating, because both have bitten real
deployments:

* **The webhook URL is never read from the config file or written to a log.**
  It is read from an environment variable named by ``alerts.webhook_url_env``,
  and every logging formatter in the project redacts webhook-shaped strings as
  a second line of defence.
* **A Discord ``?wait=true`` failure must not be treated as an alert failure.**
  When Discord's own API is degraded, the webhook can return 429 with a
  ``retry_after``, and blindly retrying produces a tight loop against a service
  that is already struggling. The notifier honours ``retry_after`` and gives up
  into the cooldown window rather than hammering.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..errors import AlertDeliveryError, SourceError
from ..models import Severity
from ..net import HttpClient

__all__ = [
    "Alert",
    "AlertDispatcher",
    "DiscordNotifier",
    "NullNotifier",
    "build_dispatcher",
    "resolve_webhook",
]

_LOG = logging.getLogger("emperor.alerts.notifier")

#: Discord's hard embed limits. Exceeding them returns 400 with a message that
#: does not say which field was wrong, so they are enforced client-side.
_EMBED_TITLE_LIMIT = 256
_EMBED_DESCRIPTION_LIMIT = 4096
_EMBED_FIELD_LIMIT = 25
_EMBED_FIELD_VALUE_LIMIT = 1024

#: Colour of the embed side bar per severity, mirroring
#: :attr:`emperor_space_tracker.models.Severity.discord_colour`.
_FALLBACK_COLOUR = 0x9BA5B4


@dataclass(frozen=True, slots=True)
class Alert:
    """A notification to deliver, independent of its transport.

    Attributes
    ----------
    rule_id
        Stable identifier of the originating rule.
    severity
        Severity ladder value.
    title
        One-line summary, used as the embed title.
    body
        Detail paragraph, used as the embed description.
    context
        Machine-readable values that triggered the alert.
    fields
        Pre-rendered ``(label, value)`` pairs for the embed's field list.
    """

    rule_id: str
    severity: Severity
    title: str
    body: str
    context: dict[str, Any] = field(default_factory=dict)
    fields: tuple[tuple[str, str], ...] = ()

    def embed(self) -> dict[str, Any]:
        """Render the alert as a Discord embed object.

        Returns
        -------
        dict[str, Any]
            A Discord-compatible embed, with every length limit already
            enforced so a long observation list cannot cause a 400.
        """
        embed: dict[str, Any] = {
            "title": self.title[:_EMBED_TITLE_LIMIT],
            "description": self.body[:_EMBED_DESCRIPTION_LIMIT],
            "color": self.severity.discord_colour or _FALLBACK_COLOUR,
        }
        if self.fields:
            embed["fields"] = [
                {
                    "name": str(name)[:_EMBED_FIELD_VALUE_LIMIT],
                    "value": str(value)[:_EMBED_FIELD_VALUE_LIMIT] or "\u200b",
                    "inline": True,
                }
                for name, value in self.fields[:_EMBED_FIELD_LIMIT]
            ]
        return embed


@runtime_checkable
class AlertDispatcher(Protocol):
    """Protocol every alert channel must satisfy."""

    def dispatch(self, alert: Alert) -> bool:
        """Deliver ``alert``.

        Parameters
        ----------
        alert
            The alert to deliver.

        Returns
        -------
        bool
            ``True`` if the channel accepted it.
        """
        ...

    def describe(self) -> str:
        """Return a human-readable description of the configured target.

        Returns
        -------
        str
        """
        ...


class NullNotifier:
    """Discards alerts after logging them at their own severity.

    This is the correct behaviour for an unconfigured node, not a stub. The
    rules still evaluate, the alerts are still persisted to the store, and they
    are still logged at the severity the engine assigned -- so an operator
    reviewing ``journalctl`` sees exactly what would have been dispatched.

    Parameters
    ----------
    log_dispatch
        Emit a line per alert at the alert's severity.

    Examples
    --------
    >>> notifier = NullNotifier(log_dispatch=False)
    >>> notifier.dispatch(Alert("r", Severity.INFO, "t", "b"))
    True
    >>> notifier.delivered_count
    1
    """

    def __init__(self, *, log_dispatch: bool = True) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.log_dispatch = log_dispatch
        self.delivered_count = 0
        self.last_alert: Alert | None = None

    def dispatch(self, alert: Alert) -> bool:
        """Log the alert and return ``True``.

        Parameters
        ----------
        alert
            The alert.

        Returns
        -------
        bool
            Always ``True``.
        """
        self.delivered_count += 1
        self.last_alert = alert
        if self.log_dispatch:
            _LOG.log(
                logging.ERROR if alert.severity.rank >= Severity.SEVERE.rank else logging.WARNING,
                "ALERT [%s] %s | %s",
                alert.severity.value.upper(),
                alert.title,
                alert.body,
            )
            if alert.fields:
                for name, value in alert.fields:
                    _LOG.debug("  %s: %s", name, value)
        return True

    def describe(self) -> str:
        """Return a description naming the disabled state.

        Returns
        -------
        str
        """
        return "null channel (alerts logged locally, nothing transmitted)"


class DiscordNotifier:
    """Posts alert embeds to a Discord webhook.

    Parameters
    ----------
    client
        Shared HTTP client.
    webhook_url
        The webhook URL. Sourced from the environment, never from a config file.
    username
        Override the webhook's default username.
    max_retries
        Total attempts per alert.
    retry_base
        Base backoff in seconds.

    Examples
    --------
    >>> from emperor_space_tracker.net import HttpClient
    >>> notifier = DiscordNotifier(HttpClient(), "https://example.invalid/hook")
    >>> notifier.dispatch(Alert("r", Severity.INFO, "t", "b"))  # doctest: +SKIP
    False
    """

    def __init__(
        self,
        client: HttpClient,
        webhook_url: str,
        *,
        username: str = "Emperor Space Tracker",
        max_retries: int = 3,
        retry_base: float = 1.5,
    ) -> None:
        """Initialise; see the class docstring for parameter semantics."""
        self.client = client
        self.webhook_url = webhook_url
        self.username = username
        self.max_retries = max(1, max_retries)
        self.retry_base = retry_base
        self.delivered_count = 0
        self.failed_count = 0

    def payload(self, alert: Alert) -> dict[str, Any]:
        """Build the webhook JSON body for an alert.

        Parameters
        ----------
        alert
            The alert.

        Returns
        -------
        dict[str, Any]
        """
        return {
            "username": self.username,
            "embeds": [alert.embed()],
            # Suppress the @everyone ping. An alert that pages every operator on
            # a fleet-wide channel for a routine warning is how a monitoring
            # system gets muted. Escalate to a mention only in the incident
            # response procedure, deliberately.
            "allowed_mentions": {"parse": []},
        }

    def dispatch(self, alert: Alert) -> bool:
        """Deliver an alert, retrying transient failures.

        Retries three classes of failure, because each of them is temporary and
        an alert that is lost is a storm nobody is paged for:

        * ``429`` -- rate limited; wait for the server's own ``retry_after``
          rather than a guessed backoff.
        * ``5xx`` -- Discord-side fault. This used to raise on the first
          non-2xx response, which meant a single blip during an event was
          indistinguishable from a permanently dead webhook and the alert was
          simply dropped.
        * transport errors -- a timeout or a reset connection.

        A ``4xx`` other than ``429`` is a permanent rejection (malformed
        payload, revoked webhook, unknown channel) and is raised immediately,
        because retrying it three times only delays the same failure.

        Parameters
        ----------
        alert
            The alert to deliver.

        Returns
        -------
        bool
            ``True`` if Discord accepted the payload.

        Raises
        ------
        AlertDeliveryError
            If every permitted attempt failed.
        """
        body = self.payload(alert)
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.post_json(self.webhook_url, body, source="discord")
            except SourceError as exc:
                message = str(exc)
                if "HTTP 429" in message:
                    delay = _parse_retry_after(message, self.retry_base * attempt)
                    _LOG.warning(
                        "discord rate limited (attempt %d/%d); standing down %.1fs",
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                if attempt >= self.max_retries:
                    self.failed_count += 1
                    raise AlertDeliveryError(message) from exc
                delay = self.retry_base * (2 ** (attempt - 1))
                _LOG.warning(
                    "discord delivery failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    message,
                    delay,
                )
                time.sleep(delay)
                continue

            if response.status in {200, 204}:
                self.delivered_count += 1
                _LOG.info("discord accepted %s", alert.rule_id)
                return True

            # 5xx is the server's problem and is worth another attempt; any
            # other non-2xx is a permanent rejection of this payload or URL.
            if 500 <= response.status < 600:
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base * (2 ** (attempt - 1))
                _LOG.warning(
                    "discord returned HTTP %d for %s (attempt %d/%d); retrying in %.1fs",
                    response.status,
                    alert.rule_id,
                    attempt,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
                continue

            self.failed_count += 1
            raise AlertDeliveryError(
                f"discord returned HTTP {response.status} for {alert.rule_id}"
            )

        self.failed_count += 1
        raise AlertDeliveryError(
            f"discord rejected {alert.rule_id} after {self.max_retries} attempts"
        )

    def describe(self) -> str:
        """Return the webhook's host and path with the secret removed.

        Returns
        -------
        str
        """
        return f"discord webhook at {_redact_url(self.webhook_url)}"


def _parse_retry_after(message: str, fallback: float) -> float:
    """Extract Discord's ``retry_after`` hint from an error message.

    Parameters
    ----------
    message
        The exception text from the HTTP layer.
    fallback
        Delay to use when no hint is present.

    Returns
    -------
    float
        Seconds to wait, clamped to a sane maximum so a hostile or misconfigured
        response cannot park the daemon for an hour.
    """
    marker = "retry_after"
    if marker in message:
        tail = message.split(marker, 1)[1]
        digits = "".join(ch for ch in tail[:12] if ch.isdigit() or ch == ".")
        try:
            return min(60.0, max(0.5, float(digits)))
        except ValueError:
            return fallback
    return min(60.0, fallback)


def _redact_url(url: str) -> str:
    """Return a webhook URL with its secret token and query string removed.

    Parameters
    ----------
    url
        The webhook URL.

    Returns
    -------
    str
        A form safe to log, e.g. ``discord.com/api/webhooks/***``.

    Examples
    --------
    >>> _redact_url("https://discord.com/api/webhooks/123456789/abcdefTOKEN?t=xyz")
    'https://discord.com/api/webhooks/***'
    """
    head, _, _ = url.partition("?")
    marker = "/webhooks/"
    if marker in head:
        prefix, _, _ = head.partition(marker)
        return f"{prefix}{marker}***"
    return "***"


def resolve_webhook(env_var: str, env: dict[str, str] | None = None) -> str | None:
    """Read the Discord webhook URL from the environment.

    Parameters
    ----------
    env_var
        Name of the environment variable to read.
    env
        Environment mapping. Defaults to :data:`os.environ`.

    Returns
    -------
    str | None
        The URL, or ``None`` if the variable is unset or does not look like a
        Discord webhook.

    Notes
    -----
    The shape is validated before use. A typo in the variable -- or a token
    pasted into the wrong one -- produces "no channel configured" rather than a
    404 on every alert for the next three months.
    """
    source = os.environ if env is None else env
    value = (source.get(env_var) or "").strip()
    if not value:
        return None
    if "discord.com/api/webhooks/" not in value and "discordapp.com/api/webhooks/" not in value:
        _LOG.error(
            "%s is set but does not look like a Discord webhook URL; "
            "expected https://discord.com/api/webhooks/<id>/<token>",
            env_var,
        )
        return None
    return value


def build_dispatcher(
    *,
    client: HttpClient,
    enabled: bool,
    webhook_url_env: str,
    max_retries: int = 3,
    env: dict[str, str] | None = None,
) -> AlertDispatcher:
    """Construct the alert channel implied by the configuration.

    Never raises. An unconfigured or misconfigured channel yields a
    :class:`NullNotifier`, because a monitoring node that refuses to start
    because nobody set an environment variable is not a monitoring node.

    Parameters
    ----------
    client
        Shared HTTP client.
    enabled
        Whether alerting is enabled at all.
    webhook_url_env
        Environment variable holding the webhook URL.
    max_retries
        Attempts per alert.
    env
        Environment mapping for injection in tests.

    Returns
    -------
    AlertDispatcher
        A :class:`DiscordNotifier` when fully configured, else a
        :class:`NullNotifier`.

    Examples
    --------
    >>> from emperor_space_tracker.net import HttpClient
    >>> build_dispatcher(client=HttpClient(), enabled=False,
    ...                  webhook_url_env="NOPE", env={}).describe()
    'null channel (alerts logged locally, nothing transmitted)'
    """
    if not enabled:
        return NullNotifier()
    url = resolve_webhook(webhook_url_env, env)
    if url is None:
        return NullNotifier()
    return DiscordNotifier(client, url, max_retries=max_retries)
