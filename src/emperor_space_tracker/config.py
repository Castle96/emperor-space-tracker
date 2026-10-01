"""Layered TOML configuration with strict validation.

Resolution order, last wins:

1. the packaged defaults (:mod:`emperor_space_tracker.data.default`),
2. ``$XDG_CONFIG_HOME/emperor-space-tracker/config.toml`` (or the platform
   equivalent) if it exists,
3. an explicit ``--config`` path,
4. ``EST_*`` environment variable overrides.

The daemon is started by systemd with a *fixed* argv, so the config path cannot
be passed positionally. It is resolved here instead, from ``$EST_CONFIG`` with an
XDG fallback. That is what makes the systemd unit distribution agnostic: the
same unit file works on Arch, Debian, RHEL and Alpine without a wrapper script.

Validation is total: :meth:`Config.validate` either returns a fully-populated,
range-checked object or raises :class:`~emperor_space_tracker.errors.ConfigError`
listing *every* problem it found, not just the first.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Final

from .errors import ConfigError
from .models import Severity

__all__ = [
    "SAR_COLLECTIONS",
    "SAR_RAW_ONLY",
    "AlertConfig",
    "ColonyConfig",
    "Config",
    "DaemonConfig",
    "LogConfig",
    "PathsConfig",
    "SarConfig",
    "SiteConfig",
    "SpaceWeatherConfig",
    "default_config_path",
    "load_config",
    "package_default_path",
    "render_default_config",
]

APP_NAME: Final = "emperor-space-tracker"
_APP_DIR: Final = "emperor-space-tracker"
_ENV_PREFIX: Final = "EST_"


def package_default_path() -> Path:
    """Return the path to the packaged ``default.toml``."""
    return Path(__file__).resolve().parent / "data" / "default.toml"


def default_config_path() -> Path:
    """Return the user override path, honouring ``$XDG_CONFIG_HOME``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / _APP_DIR / "config.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    """Read and parse a TOML file, converting failures to :class:`ConfigError`."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        msg = f"cannot read {path}: {exc.strerror or exc}"
        raise ConfigError(msg) from exc
    try:
        parsed: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        msg = f"{path} is not valid UTF-8: {exc}"
        raise ConfigError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path} is not valid TOML: {exc}"
        raise ConfigError(msg) from exc
    return parsed


def _coerce(raw: Any, *, field_name: str, where: str) -> Any:
    """Coerce a TOML scalar to the type declared by its dataclass field.

    TOML cannot express ``None``, so an absent optional is indistinguishable
    from an explicit zero. Strings are still accepted for numeric fields
    because hand-edited configs routinely quote numbers.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
    return raw


def _build_section(
    section_cls: type[Any],
    data: dict[str, Any],
    *,
    where: str,
) -> Any:
    """Instantiate ``section_cls`` from a mapping, rejecting unknown keys.

    Unknown keys are a hard error rather than a warning. A silently ignored
    ``storm_kp = 7.0`` typo in a field config is exactly the kind of failure
    that only surfaces at 3 a.m. in the middle of a storm.
    """
    known = {f.name: f for f in fields(section_cls)}
    unknown = set(data) - set(known)
    if unknown:
        msg = f"unknown key(s) in [{where}]: {', '.join(sorted(unknown))}"
        raise ConfigError(msg)

    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value: Any = _coerce(data[name], field_name=name, where=where)
        # A dataclass field with no default holds MISSING, not None, so the
        # test for "no fallback available" has to compare against MISSING.
        # Checking `f.default is None` instead silently inverted the branch:
        # it matched only fields whose default *is* None and let every
        # genuinely required field fall through to a TypeError.
        required = f.default is MISSING and f.default_factory is MISSING
        if value is None and required:
            continue
        kwargs[name] = value

    try:
        return section_cls(**kwargs)
    except TypeError as exc:
        msg = f"[{where}] has a value of the wrong type: {exc}"
        raise ConfigError(msg) from exc


def _check(name: str, value: float, low: float, high: float, problems: list[str]) -> None:
    """Append a range problem to ``problems`` if ``value`` is out of bounds."""
    if not low <= value <= high:
        problems.append(f"{name}={value!r} is outside the supported range [{low}, {high}]")


@dataclass(slots=True)
class SiteConfig:
    """Identity and position of the physical monitoring node."""

    name: str = "unnamed-node"
    timezone: str = "UTC"
    latitude: float = -77.85
    longitude: float = 166.67


@dataclass(slots=True)
class PathsConfig:
    """Where mutable state is written."""

    state_dir: str = "~/.local/state/emperor-space-tracker"
    max_db_mib: int = 64
    retention_days: int = 30

    def resolved_state_dir(self) -> Path:
        """Return the state directory with ``~`` expanded and parents created."""
        return Path(self.state_dir).expanduser()

    def ensure(self) -> Path:
        """Create the state directory if absent and return it."""
        path = self.resolved_state_dir()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"cannot create state directory {path}: {exc.strerror or exc}"
            raise ConfigError(msg) from exc
        return path

    @property
    def database_path(self) -> Path:
        """Return the SQLite database path."""
        return self.resolved_state_dir() / "tracker.sqlite3"

    @property
    def log_path(self) -> Path:
        """Return the rotating log file path."""
        return self.resolved_state_dir() / "tracker.log"


@dataclass(slots=True)
class DaemonConfig:
    """Supervision loop tuning and the resource ceiling."""

    poll_interval_seconds: float = 300.0
    http_timeout_seconds: float = 20.0
    max_rss_bytes: int = 48 * 1024 * 1024
    rss_warn_fraction: float = 0.85
    consecutive_failures_before_page: int = 3
    heartbeat_seconds: float = 3600.0
    unhealthy_repeat_seconds: float = 3600.0


@dataclass(slots=True)
class SpaceWeatherConfig:
    """NOAA SWPC ingestion and geomagnetic threshold settings."""

    enabled: bool = True
    storm_kp: float = 5.0
    storm_kp_clear: float = 4.0
    high_wind_kms: float = 700.0
    southward_bz_nt: float = -5.0
    consecutive_samples: int = 2
    max_sample_age_seconds: float = 3600.0
    track_f107: bool = True


#: Which Earth Engine collection carries which Sentinel-1 polarisation.
#:
#: This mapping is the whole reason ``polarisation`` is a separate setting. The
#: GRD products are dual-pol VV+VH and contain no HH at all, so asking GRD for
#: HH is not a slow query, it is an empty band selection. HH exists only in the
#: RAW product, which arrives uncalibrated and unfiltered. Selecting the
#: collection from the requested band means no configuration can name a band the
#: chosen dataset does not carry.
SAR_COLLECTIONS: Final[dict[str, str]] = {
    "VV": "COPERNICUS/S1_GRD",
    "VH": "COPERNICUS/S1_GRD",
    "HH": "COPERNICUS/S1_RAW",
    "HV": "COPERNICUS/S1_RAW",
}

#: Bands only RAW carries, and therefore bands that need calibration work before
#: their dB values mean anything.
SAR_RAW_ONLY: Final = frozenset({"HH", "HV"})


@dataclass(slots=True)
class SarConfig:
    """Sentinel-1 fast-ice backscatter settings.

    ``polarisation`` also selects the Earth Engine collection: VV/VH read from
    the calibrated, speckle-filtered GRD product, while HH/HV require the RAW
    product and the extra processing noted in :data:`SAR_RAW_ONLY`.
    """

    enabled: bool = True
    backend: str = "synthetic"
    platform: str = "SENTINEL-1"
    orbit_type: str = "ASCENDING"
    polarisation: str = "VV"
    radius_meters: int = 5000
    resolution_meters: int = 100
    lookback_days: int = 30
    revisit_hours: float = 12.0
    grid_cells: int = 41
    synthetic_seed: int = 0
    polynya_window_scenes: int = 4
    polynya_min_drop_db: float = 1.5
    polynya_clear_drop_db: float = 0.5

    def polynya_drop_band(self) -> tuple[float, float]:
        """Return the ``(enter, clear)`` polynya decline thresholds in dB.

        Both are expressed as positive magnitudes of decline and returned as
        negative values, which is the sign convention the trend detector uses.

        The two must be separated or the rule chatters exactly at the threshold
        it exists to detect. ``clear`` is the shallower of the pair: a trend
        recovers gradually, so it is still worth reporting while it is shallower
        than the enter threshold but steeper than the clear threshold.
        """
        enter = -abs(self.polynya_min_drop_db)
        clear = -abs(self.polynya_clear_drop_db)
        return enter, clear


@dataclass(slots=True)
class ColonyConfig:
    """GBIF and SCAR biological census settings."""

    species: str = "Aptenodytes forsteri"
    max_distance_km: float = 700.0
    max_colonies: int = 12
    refresh_every_polls: int = 60
    include_gbif: bool = True


@dataclass(slots=True)
class AlertConfig:
    """Outbound notification channel settings."""

    enabled: bool = True
    webhook_url_env: str = "EMPEROR_DISCORD_WEBHOOK"
    min_severity: str = "info"
    cooldown_seconds: float = 1800.0
    post_quiet_heartbeat: bool = False
    max_retries: int = 3

    def severity_floor(self) -> Severity:
        """Return :attr:`min_severity` as a validated :class:`Severity`."""
        try:
            return Severity(self.min_severity.strip().lower())
        except ValueError as exc:
            valid = ", ".join(s.value for s in Severity)
            msg = f"alerts.min_severity={self.min_severity!r} is not one of: {valid}"
            raise ConfigError(msg) from exc


@dataclass(slots=True)
class DashboardAuthConfig:
    """Authentication for the dashboard.

    Two mechanisms, because they solve different halves of the problem.

    ``proxy_secret`` is the app's half of a reverse-proxy arrangement. The proxy
    terminates TLS, verifies the human's credential on the connection, and
    forwards the resulting identity in a header signed with this shared secret.
    The app believes that header only when the signature matches, so a host
    that reaches the port directly cannot forge an identity by guessing a
    username. Verified in
    :func:`emperor_space_tracker.auth.verify_identity`, which fails closed.

    ``password_fallback`` is for the case where no proxy is in front. A login
    form in the dashboard itself, backed by :class:`~emperor_space_tracker.auth.AuthStore`.
    Streamlit 1.64 removed ``[server] password`` and gives scripts no cookie
    setter, so there is no supported way to keep a session across a browser
    refresh; this is deliberately not offered as a first-class path for that
    reason, and the comment at :func:`verify_identity` explains the arrangement
    that is.
    """

    enabled: bool = False
    # The *name* of the variable holding the secret, never a secret.
    proxy_secret_env: str = "EST_DASHBOARD_PROXY_SECRET"  # noqa: S105
    max_failures: int = 5
    lockout_base_seconds: float = 60.0
    lockout_cap_seconds: float = 900.0


@dataclass(slots=True)
class DashboardConfig:
    """Streamlit frontend settings.

    These are the values ``est install`` bakes into the dashboard unit, so a
    deployment that should come up on boot with a particular bind address and
    port records that intent in the config file rather than in a shell history.

    ``host`` defaults to loopback. A wider bind is only coherent with
    ``auth.enabled``; :meth:`Config.validate` refuses the combination rather
    than warning, because "reachable from the network with no authentication"
    is the exact state this project has been trying to avoid. The right way to
    get a remote dashboard is a reverse proxy that authenticates the connection
    and forwards a signed identity header -- see :class:`DashboardAuthConfig`.
    """

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8501
    headless: bool = True
    max_rss_bytes: int = 512 * 1024 * 1024
    auth: DashboardAuthConfig = field(default_factory=DashboardAuthConfig)

    @property
    def bind_is_not_loopback(self) -> bool:
        """Return whether this bind address is reachable off the node.

        A wildcard or non-loopback address exposes the dashboard to every host
        that can route to it, which is why it is gated on authentication.
        """
        return self.host not in {"127.0.0.1", "::1", "localhost"}


@dataclass(slots=True)
class LogConfig:
    """Logging destination and verbosity."""

    level: str = "INFO"
    max_bytes: int = 2 * 1024 * 1024
    backup_count: int = 3
    json_file: bool = False


@dataclass(slots=True)
class Config:
    """The fully-resolved configuration for one tracker process."""

    site: SiteConfig = field(default_factory=SiteConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    space_weather: SpaceWeatherConfig = field(default_factory=SpaceWeatherConfig)
    sar: SarConfig = field(default_factory=SarConfig)
    colonies: ColonyConfig = field(default_factory=ColonyConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LogConfig = field(default_factory=LogConfig)

    #: Absolute path this config was loaded from, for diagnostics.
    source_path: Path | None = None
    #: Non-fatal notices raised while layering the config files.
    warnings: tuple[str, ...] = ()

    def validate(self) -> Config:
        """Range-check every section, raising with *all* problems found.

        Returns
        -------
        Config
            ``self``, so this can be used as the tail of a loader chain.

        Raises
        ------
        ConfigError
            If any value is out of range or internally inconsistent.
        """
        problems: list[str] = []

        _check("site.latitude", self.site.latitude, -90.0, 90.0, problems)
        _check("site.longitude", self.site.longitude, -180.0, 180.0, problems)
        if not self.site.name.strip():
            problems.append("site.name must not be empty")

        _check(
            "daemon.poll_interval_seconds",
            self.daemon.poll_interval_seconds,
            30.0,
            86400.0,
            problems,
        )
        _check(
            "daemon.http_timeout_seconds", self.daemon.http_timeout_seconds, 1.0, 300.0, problems
        )
        _check(
            "daemon.max_rss_bytes",
            self.daemon.max_rss_bytes,
            8 * 1024 * 1024,
            2 * 1024**3,
            problems,
        )
        _check("daemon.heartbeat_seconds", self.daemon.heartbeat_seconds, 60.0, 86400.0, problems)
        _check(
            "daemon.unhealthy_repeat_seconds",
            self.daemon.unhealthy_repeat_seconds,
            0.0,
            86400.0,
            problems,
        )
        if not 0.1 <= self.daemon.rss_warn_fraction <= 1.0:
            problems.append("daemon.rss_warn_fraction must be between 0.1 and 1.0")
        if self.daemon.consecutive_failures_before_page < 1:
            problems.append("daemon.consecutive_failures_before_page must be >= 1")

        if self.space_weather.storm_kp < 0.0 or self.space_weather.storm_kp > 9.0:
            problems.append(
                f"space_weather.storm_kp={self.space_weather.storm_kp} outside Kp scale 0..9"
            )
        if self.space_weather.storm_kp_clear >= self.space_weather.storm_kp:
            problems.append(
                "space_weather.storm_kp_clear "
                f"({self.space_weather.storm_kp_clear}) must be below storm_kp "
                f"({self.space_weather.storm_kp}); the gap is the anti-flapping hysteresis"
            )
        _check(
            "space_weather.high_wind_kms", self.space_weather.high_wind_kms, 200.0, 2000.0, problems
        )
        _check(
            "space_weather.southward_bz_nt",
            self.space_weather.southward_bz_nt,
            -100.0,
            0.0,
            problems,
        )
        if self.space_weather.consecutive_samples < 1:
            problems.append("space_weather.consecutive_samples must be >= 1")

        if self.sar.backend not in {"synthetic", "gee"}:
            problems.append(f"sar.backend={self.sar.backend!r} must be 'synthetic' or 'gee'")
        band = self.sar.polarisation.upper()
        if band not in SAR_COLLECTIONS:
            allowed = ", ".join(sorted(SAR_COLLECTIONS))
            problems.append(
                f"sar.polarisation={self.sar.polarisation!r} must be one of {allowed}"
            )
        _check("sar.radius_meters", self.sar.radius_meters, 100.0, 100_000.0, problems)
        _check("sar.resolution_meters", self.sar.resolution_meters, 10.0, 1000.0, problems)
        _check("sar.lookback_days", self.sar.lookback_days, 1.0, 3650.0, problems)
        _check("sar.revisit_hours", self.sar.revisit_hours, 1.0, 336.0, problems)
        if self.sar.grid_cells % 2 != 1:
            problems.append("sar.grid_cells must be odd so the grid is centred on the colony")
        if not 5 <= self.sar.grid_cells <= 501:
            problems.append("sar.grid_cells must be between 5 and 501")
        span = self.sar.grid_cells * self.sar.resolution_meters
        if span > 2 * self.sar.radius_meters:
            problems.append(
                f"sar.grid_cells x resolution ({span} m) spans more than the "
                f"2 x radius window ({2 * self.sar.radius_meters} m)"
            )
        _check("sar.polynya_window_scenes", self.sar.polynya_window_scenes, 2.0, 60.0, problems)
        _check("sar.polynya_min_drop_db", self.sar.polynya_min_drop_db, 0.1, 20.0, problems)
        _check("sar.polynya_clear_drop_db", self.sar.polynya_clear_drop_db, 0.0, 20.0, problems)
        if self.sar.polynya_clear_drop_db >= self.sar.polynya_min_drop_db:
            problems.append(
                f"sar.polynya_clear_drop_db ({self.sar.polynya_clear_drop_db}) must be "
                f"less than polynya_min_drop_db ({self.sar.polynya_min_drop_db}); the "
                "gap between them is the anti-flapping hysteresis band"
            )
        if self.sar.polynya_window_scenes > self.sar.lookback_days * 2:
            # Sentinel-1 repeat is ~12 h where covered and days where not, so a
            # window longer than the lookback can never be filled from one pass.
            problems.append(
                f"sar.polynya_window_scenes ({self.sar.polynya_window_scenes}) needs more "
                f"acquisitions than a {self.sar.lookback_days}-day lookback can hold"
            )

        _check("colonies.max_distance_km", self.colonies.max_distance_km, 1.0, 20_000.0, problems)
        if self.colonies.max_colonies < 1:
            problems.append("colonies.max_colonies must be >= 1")
        if not self.colonies.species.strip():
            problems.append("colonies.species must not be empty")

        _check("dashboard.port", self.dashboard.port, 1.0, 65535.0, problems)
        _check(
            "dashboard.max_rss_bytes",
            self.dashboard.max_rss_bytes,
            64 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            problems,
        )
        if not self.dashboard.host.strip():
            problems.append("dashboard.host must not be empty")
        if self.dashboard.bind_is_not_loopback and self.dashboard.headless is False:
            # Not an error -- opening a browser is legitimate on a desktop node
            # -- but it is the combination that surprises people, so the
            # install path warns about it rather than the validator rejecting
            # a working configuration.
            self.warnings = (
                *self.warnings,
                "dashboard.host is not loopback and dashboard.headless is false: a "
                "browser will be opened on start, which is meaningless for a "
                "systemd unit with no display",
            )

        auth = self.dashboard.auth
        _check("dashboard.auth.max_failures", auth.max_failures, 1.0, 100.0, problems)
        _check(
            "dashboard.auth.lockout_base_seconds",
            auth.lockout_base_seconds,
            1.0,
            3600.0,
            problems,
        )
        _check(
            "dashboard.auth.lockout_cap_seconds",
            auth.lockout_cap_seconds,
            auth.lockout_base_seconds,
            86_400.0,
            problems,
        )
        if not auth.proxy_secret_env.strip():
            problems.append("dashboard.auth.proxy_secret_env must name an environment variable")

        # The fail-closed check. A dashboard reachable from off the machine with
        # authentication switched off is the exact state this project has been
        # trying to avoid, and it is silent: nothing about the rendered page
        # would tell an operator the store is world-readable. So it is a hard
        # error, not a warning.
        #
        # The one exception is a bind that is not a wildcard and not loopback --
        # a Tailscale address, say. Those are still refused, because a private
        # address is only private for as long as nobody adds a route.
        if self.dashboard.enabled and self.dashboard.bind_is_not_loopback and not auth.enabled:
            problems.append(
                f"dashboard.host={self.dashboard.host!r} makes the dashboard reachable off "
                f"this machine, but dashboard.auth.enabled is false. Anyone who can route "
                f"to this port would be able to read the store -- colony positions, SAR "
                f"grids and alerts. Either set dashboard.auth.enabled = true, or bind "
                f"127.0.0.1 and reach the dashboard through a reverse proxy that "
                f"authenticates the connection."
            )

        _check("alerts.cooldown_seconds", self.alerts.cooldown_seconds, 0.0, 86_400.0, problems)
        if self.alerts.max_retries < 1:
            problems.append("alerts.max_retries must be >= 1")
        if not self.alerts.webhook_url_env.strip():
            problems.append("alerts.webhook_url_env must name a non-empty environment variable")
        try:
            self.alerts.severity_floor()
        except ConfigError as exc:
            problems.append(str(exc))

        _check("paths.max_db_mib", self.paths.max_db_mib, 0.0, 65_536.0, problems)
        if not 1 <= self.paths.retention_days <= 3650:
            problems.append("paths.retention_days must be between 1 and 3650")
        if self.logging.max_bytes < 4096:
            problems.append("logging.max_bytes must be >= 4096")
        if not 0 <= self.logging.backup_count <= 64:
            problems.append("logging.backup_count must be between 0 and 64")
        if self.logging.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            problems.append(f"logging.level={self.logging.level!r} is not a valid level name")

        if problems:
            bullet = "\n  - "
            joined = bullet.join(problems)
            msg = f"configuration is invalid:\n  - {joined}"
            raise ConfigError(msg)

        return self

    def effective_rss_mib(self) -> float:
        """Return the configured RSS ceiling in MiB, for display."""
        return self.daemon.max_rss_bytes / (1024 * 1024)

    def describe(self) -> str:
        """Return a redacted, human-readable rendering of the config.

        The Discord webhook is never interpolated here; only the *name* of the
        environment variable that holds it is shown.
        """
        lines = [
            f"config source   : {self.source_path or '<packaged defaults>'}",
            f"site            : {self.site.name} @ {self.site.latitude:.3f}, "
            f"{self.site.longitude:.3f} ({self.site.timezone})",
            f"state dir       : {self.paths.resolved_state_dir()}",
            f"database        : {self.paths.database_path}",
            f"log file        : {self.paths.log_path}",
            f"poll interval   : {self.daemon.poll_interval_seconds:g}s",
            f"rss ceiling     : {self.effective_rss_mib():.1f} MiB",
            f"space weather   : {'on' if self.space_weather.enabled else 'off'} "
            f"(storm at Kp >= {self.space_weather.storm_kp:g}, "
            f"clears at Kp <= {self.space_weather.storm_kp_clear:g})",
            f"sar             : {'on' if self.sar.enabled else 'off'} "
            f"backend={self.sar.backend} "
            f"pol={self.sar.polarisation.upper()}"
            f"({SAR_COLLECTIONS.get(self.sar.polarisation.upper(), 'n/a').split('/')[-1]}) "
            f"grid={self.sar.grid_cells}x{self.sar.grid_cells}@{self.sar.resolution_meters:g}m "
            f"revisit={self.sar.revisit_hours:g}h",
            f"colonies        : {self.colonies.species}, "
            f"max {self.colonies.max_colonies} within {self.colonies.max_distance_km:g} km",
            f"alerts          : {'on' if self.alerts.enabled else 'off'} "
            f"floor={self.alerts.min_severity} cooldown={self.alerts.cooldown_seconds:g}s "
            f"webhook=${self.alerts.webhook_url_env}",
            f"dashboard       : {'on' if self.dashboard.enabled else 'off'} "
            f"http://{self.dashboard.host}:{self.dashboard.port} "
            f"ceiling={self.dashboard.max_rss_bytes / (1024 * 1024):.0f} MiB "
            f"auth={'on' if self.dashboard.auth.enabled else 'OFF'}",
            f"logging         : {self.logging.level.upper()} -> {self.paths.log_path}",
        ]
        for warning in self.warnings:
            lines.append(f"warning         : {warning}")
        return "\n".join(lines)


def _env_overrides(env: dict[str, str]) -> dict[str, Any]:
    """Translate ``EST_``-prefixed environment variables into config layers.

    Supported forms::

        EST_SITE__LATITUDE=-78.1                    -> site.latitude (float)
        EST_DAEMON__POLL_INTERVAL_SECONDS=600       -> int
        EST_ALERTS__MIN_SEVERITY=warning            -> str
        EST_COLONIES__SPECIES="Pygoscelis kerguelensis"  -> str

    Values are parsed as TOML scalars first, so ``-1`` becomes an int, ``true`` a
    bool and ``8501`` an int rather than a string. Anything TOML cannot parse
    falls back to the raw string, which is what makes the unquoted forms above
    work at all: a bare word is not a TOML scalar, so without the fallback every
    string-valued setting -- every severity level, every timezone, and any
    binomial species name, which always contains a space -- would have to be
    quoted in the environment.

    A value that parses as something nonsensical for its key is still reported
    later by :meth:`Config.validate`, naming the setting; the fallback is only
    about not rejecting a perfectly good word.
    """
    layers: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith(_ENV_PREFIX) or key == "EST_CONFIG":
            continue
        remainder = key[len(_ENV_PREFIX) :]
        if "__" not in remainder:
            continue
        section, _, name = remainder.partition("__")
        section = section.lower()
        name = name.lower()
        try:
            parsed: Any = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            # Not a TOML scalar. A bare word is far more likely to be intended
            # as a string than as a mistake, so take it literally and let
            # validation judge the result.
            parsed = value
        layers.setdefault(section, {})[name] = parsed
    return layers


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto a copy of ``base``."""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def render_default_config() -> str:
    """Return the contents of the packaged default configuration file."""
    return package_default_path().read_text(encoding="utf-8")


def load_config(
    config_path: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
    use_user_config: bool = True,
) -> Config:
    """Load, layer and validate the configuration.

    Parameters
    ----------
    config_path
        Explicit TOML path. When given, it is layered *on top of* the user
        config and the packaged defaults.
    env
        Environment mapping; defaults to :data:`os.environ`. Injectable so tests
        never touch the real process environment.
    use_user_config
        Set ``False`` to ignore ``$XDG_CONFIG_HOME`` entirely. The packaged
        defaults alone are then the baseline.

    Returns
    -------
    Config
        A validated configuration.

    Raises
    ------
    ConfigError
        If any layer is unparseable, contains unknown keys, or the merged
        result fails validation.

    Examples
    --------
    >>> cfg = load_config(use_user_config=False)  # doctest: +SKIP
    >>> cfg.site.name  # doctest: +SKIP
    'Ross Sea Node'
    """
    environ = dict(os.environ if env is None else env)
    warnings: list[str] = []

    merged: dict[str, Any] = _read_toml(package_default_path())
    origin = package_default_path()

    if config_path is None:
        from_env = environ.get("EST_CONFIG")
        if from_env:
            candidate = Path(from_env).expanduser()
            if not candidate.is_file():
                msg = f"EST_CONFIG points at {candidate}, which does not exist"
                raise ConfigError(msg)
            config_path = candidate

    user_path: Path | None = None
    if use_user_config and config_path is None:
        user_path = default_config_path()
        if user_path.is_file():
            config_path = user_path

    if config_path is not None:
        explicit = Path(config_path).expanduser()
        if not explicit.is_file():
            msg = f"config file {explicit} does not exist"
            raise ConfigError(msg)
        merged = _merge(merged, _read_toml(explicit))
        origin = explicit
        if user_path is not None and explicit != user_path:
            warnings.append(
                f"loaded {explicit} on top of {user_path}; both are active"
            )

    env_layers = _env_overrides(environ)
    if env_layers:
        merged = _merge(merged, env_layers)
        warnings.append(f"applied {sum(len(v) for v in env_layers.values())} EST_* override(s)")

    section_map: dict[str, type[Any]] = {
        "site": SiteConfig,
        "paths": PathsConfig,
        "daemon": DaemonConfig,
        "space_weather": SpaceWeatherConfig,
        "sar": SarConfig,
        "colonies": ColonyConfig,
        "alerts": AlertConfig,
        "dashboard": DashboardConfig,
        "dashboard.auth": DashboardAuthConfig,
        "logging": LogConfig,
    }
    unknown_sections = set(merged) - set(section_map)
    if unknown_sections:
        msg = f"unknown section(s) in {origin}: {', '.join(sorted(unknown_sections))}"
        raise ConfigError(msg)

    # A dotted name in `section_map` denotes a nested dataclass, so
    # `[dashboard.auth]` is read as `merged["dashboard"]["auth"]` and becomes
    # `Config(dashboard=DashboardConfig(auth=...))`.
    #
    # Children are assembled before their parent, for two reasons. The parent
    # must not be handed its child's key: `_build_section` would either reject
    # it as unknown or pass the raw dict straight through, leaving a dict where
    # a dataclass belongs. And a parent already constructed cannot be merged
    # into by dict update, because it is an object rather than a mapping.
    children_of: dict[str, list[str]] = {}
    for name in section_map:
        parent, _, child = name.rpartition(".")
        if parent:
            children_of.setdefault(parent, []).append(child)

    def _raw_for(name: str) -> dict[str, Any]:
        """Return the TOML data for a possibly-dotted section name."""
        head, _, tail = name.rpartition(".")
        if not head:
            top = merged.get(name, {})
            return top if isinstance(top, dict) else {}
        parent = merged.get(head, {})
        # A malformed `[a]` that is not a table cannot hold a child section;
        # report it through the same unknown-section path as any other mistake.
        nested: Any = parent.get(tail, {}) if isinstance(parent, dict) else {}
        if not isinstance(nested, dict):
            return {}
        return dict(nested)

    def _build(name: str) -> Any:
        """Construct one section, adopting any already-built children."""
        data = dict(_raw_for(name))
        for child in children_of.get(name, ()):
            # The child owns this key now; passing it through as well would make
            # the parent hold a dict *and* a dataclass under the same name.
            data.pop(child, None)
        built_section = _build_section(
            section_map[name], data, where=f"{origin.name}:[{name}]"
        )
        for child in children_of.get(name, ()):
            setattr(built_section, child, _build(f"{name}.{child}"))
        return built_section

    built = {name: _build(name) for name in section_map if "." not in name}
    config = Config(**built, source_path=origin, warnings=tuple(warnings))
    return config.validate()


def save_user_config(path: Path | None = None, *, overwrite: bool = False) -> Path:
    """Write the packaged defaults to the user config location.

    Parameters
    ----------
    path
        Destination. Defaults to :func:`default_config_path`.
    overwrite
        Required to be ``True`` if the destination already exists; the
        function refuses to clobber a field-tuned config by default.

    Returns
    -------
    Path
        The path written.
    """
    target = path if path is not None else default_config_path()
    if target.exists() and not overwrite:
        msg = f"{target} already exists; pass overwrite=True to replace it"
        raise ConfigError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_default_config(), encoding="utf-8")
    return target
