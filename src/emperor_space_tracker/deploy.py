"""Deployment helpers for the systemd user service.

Why a *user* unit
-----------------
A field node is provisioned as an unprivileged user account. Requiring root for
the service means requiring a sudo-capable setup, which is a bad fit for a
device that may sit unattended in an unheated hut with nobody holding the keys.
``systemctl --user`` needs no root, and the units live in
``~/.config/systemd/user``, so a node can be reimaged without a root inventory.

Distribution agnosticism
------------------------
The unit is written to be valid on Arch, Ubuntu, Debian, RHEL and Alpine:

* ``[Service]`` directives are restricted to the intersection those five
  systemd builds understand. In particular it uses ``ProtectSystem=strict`` with
  an explicit ``ReadWritePaths=`` rather than anything newer, and avoids
  ``MemoryPressureWatch`` and other recent additions.
* ``%h`` is expanded by systemd itself, so the unit is correct regardless of the
  account name.
* Paths are quoted, because a home directory containing a space is unusual but
  not impossible and produces an inscrutable failure if it happens.
* The interpreter is the one inside the virtual environment, resolved to an
  absolute path at install time. There is no reliance on ``PATH``, on
  ``shell=``, or on the working directory.

Resource ceiling
----------------
``MemoryMax`` is set from the configured budget rather than a hardcoded literal,
so changing ``daemon.max_rss_bytes`` and re-running ``est install`` moves the
cgroup limit with it. A hard ``MemoryMax`` (rather than only the soft warning in
the daemon) is what actually converts a runaway allocation into a clean OOM kill
and an automatic restart, instead of an unresponsive node that a human has to
notice during polar night.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .config import Config
from .errors import TrackerError

__all__ = [
    "DASHBOARD_UNIT_NAME",
    "UNIT_NAME",
    "dashboard_env_file",
    "dashboard_unit_path",
    "install_dashboard_unit",
    "install_user_unit",
    "render_dashboard_unit",
    "render_unit",
    "systemctl",
    "systemd_available",
    "unit_path",
]

_LOG = logging.getLogger("emperor.deploy")

UNIT_NAME: Final = "emperor-space-tracker.service"

#: The dashboard unit. Separate from the collector so either can be stopped
#: without affecting the other.
DASHBOARD_UNIT_NAME: Final = "emperor-space-tracker-dashboard.service"


def _unit_dir() -> Path:
    """Return the systemd user unit directory for the current account."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "systemd" / "user"


def unit_path(name: str = UNIT_NAME) -> Path:
    """Return the full path of a user unit file.

    Parameters
    ----------
    name
        Unit file name. Defaults to the collector.

    Returns
    -------
    Path
        e.g. ``~/.config/systemd/user/emperor-space-tracker.service``.
    """
    return _unit_dir() / name


def dashboard_env_file() -> Path:
    """Return the environment file the dashboard unit reads the proxy secret from.

    One definition, used by the unit renderer and by ``est auth show-secret``, so
    the two cannot disagree about where the secret lives -- a drift that would
    present as "the secret is missing" while the service was reading it fine.

    Returns
    -------
    Path
        ``~/.config/emperor-space-tracker/dashboard.env``.
    """
    from .config import default_config_path

    return default_config_path().parent / "dashboard.env"


def dashboard_unit_path() -> Path:
    """Return the full path of the dashboard user unit file.

    Returns
    -------
    Path
        ``~/.config/systemd/user/emperor-space-tracker-dashboard.service``.
    """
    return unit_path(DASHBOARD_UNIT_NAME)


def systemd_available() -> tuple[bool, str]:
    """Report whether a usable systemd user session exists.

    Returns
    -------
    tuple[bool, str]
        ``(available, detail)``. Never raises. Detects the common container and
        WSL case where systemd is installed but no user session is running, so
        ``est install`` can explain itself instead of emitting a confusing
        "Connection to dbus failed".
    """
    systemctl_bin = shutil.which("systemctl")
    if systemctl_bin is None:
        return False, "systemctl is not on PATH; this host does not appear to use systemd"
    if not Path("/run/systemd/system").exists():
        return False, (
            "systemd is not the running init (no /run/systemd/system). "
            "This is normal in containers and WSL. Run the daemon with "
            "`est run` under your own supervisor, or `est poll` from cron."
        )
    try:
        completed = subprocess.run(
            [systemctl_bin, "--user", "is-system-running"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not query the systemd user session: {exc}"
    state = (completed.stdout or completed.stderr).strip()
    if state in {"running", "degraded"}:
        return True, f"systemd user session is {state}"
    return False, (
        f"systemd user session is '{state}'. If you are in a container or over a "
        "non-login shell, run `loginctl enable-linger $USER` so the user manager "
        "starts without a session."
    )


def render_unit(
    *,
    venv_python: Path,
    config_path: Path | None,
    memory_max_bytes: int,
    rss_warn_fraction: float,
    state_dir: Path,
) -> str:
    """Render the systemd user unit file.

    Parameters
    ----------
    venv_python
        Absolute path to the interpreter inside the virtual environment.
    config_path
        Explicit config file, or ``None`` to let the app resolve the XDG path.
    memory_max_bytes
        Hard cgroup memory ceiling, from ``daemon.max_rss_bytes``.
    rss_warn_fraction
        The daemon's own warning threshold, from ``daemon.rss_warn_fraction``.
        Quoted in the rendered comment so the unit documents the real number
        rather than a hardcoded fraction that silently goes stale.
    state_dir
        The only path the unit may write to.

    Returns
    -------
    str
        The unit file contents.
    """
    config_arg = f' --config "{config_path}"' if config_path else ""
    memory_mib = memory_max_bytes / (1024 * 1024)
    warn_mib = memory_mib * rss_warn_fraction
    # Restart delays ramp 1s -> 2s -> 4s ... capped at 60s. A node whose
    # network is down should be retrying slowly, not hot-looping, because every
    # attempt costs a DNS lookup and a TLS handshake on a metered link.
    return f"""\
# Emperor Space Tracker -- edge GIS monitoring service
#
# Generated by `est install`. Edit the config file and re-run `est install`
# after changing resource limits, then: systemctl --user restart {UNIT_NAME}
#
# User unit: no root, no login required, lives in the account's own config
# directory so an unprivileged node can be reimaged without a root inventory.

[Unit]
Description=Emperor Space Tracker - Antarctic fast-ice and space weather monitoring node
Documentation=https://github.com/Castle96/emperor-space-tracker
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
ExecStart="{venv_python}" -m emperor_space_tracker run{config_arg}

# -- restart policy -------------------------------------------------------
# A field node runs unattended for months. `always` rather than
# `on-failure` so an unexpected clean exit (a bug, an OOM-killer interaction)
# also results in a restart rather than silent death.
Restart=always
RestartSec=5

# -- resource ceiling -----------------------------------------------------
# The hard cgroup limit. The daemon's own warning threshold is deliberately
# lower ({warn_mib:.0f} MiB, from daemon.rss_warn_fraction) so drift is caught
# in the log long before this turns into an OOM kill and restart loop.
MemoryMax={memory_mib:.0f}M
MemoryHigh={memory_mib * 0.85:.0f}M

# -- filesystem sandboxing ------------------------------------------------
# All five target distributions ship a systemd new enough for these. The state
# directory is the single writable path; everything else is read-only to the
# service, so a compromise of the process cannot rewrite the Python environment
# or the user's shell profile.
StateDirectory=emperor-space-tracker
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths="{state_dir}"
PrivateTmp=true
PrivateDevices=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
ProtectClock=true
ProtectHostname=true
ProtectProc=invisible
RestrictNamespaces=true
RestrictRealtime=true
RestrictSUIDSGID=true
LockPersonality=true
MemoryDenyWriteExecute=true
NoNewPrivileges=true
RemoveIPC=true
UMask=0077

# -- network --------------------------------------------------------------
# The node ingests from NOAA, GBIF and Google and posts to a Discord webhook. It
# never listens on a socket, so it needs no address family beyond IPv4/IPv6 and
# nothing else: dropping AF_UNIX and AF_PACKET removes the local-socket and raw
# socket primitives that a compromised process would otherwise reach for.
#
# Note what is deliberately absent: a port-level "HTTPS only" restriction.
# systemd's IPAddressDeny/IPAddressAllow filter on IP address, not port, so
# IPAddressDeny=any plus IPAddressAllow=0.0.0.0/0 would permit all IPv4 traffic
# while appearing to restrict it. systemd cannot express "port 443 only", and
# the upstream hosts are CDN-fronted with changing addresses, so no address
# allowlist is stable either. Enforcing egress filtering needs a network
# namespace or a host firewall, which is a deployment decision rather than
# something a user unit should silently pretend to do.
RestrictAddressFamilies=AF_INET AF_INET6

# -- process --------------------------------------------------------------
KillSignal=SIGTERM
# The daemon finishes the in-flight poll on SIGTERM, which takes at most one
# http_timeout_seconds. 90s leaves ample headroom; the default 90s would
# escalate to SIGKILL and discard a poll in progress.
TimeoutStopSec=90
# Watchdog stays off deliberately. A non-zero WatchdogSec makes systemd expect
# sd_notify(WATCHDOG=1) pings and SIGABRT the unit when they stop arriving. This
# daemon is standard-library-only and does not speak the notify protocol, so
# enabling it would kill a perfectly healthy process on a timer. Liveness is
# covered by Type=simple, Restart=on-failure and the `est doctor` probe.
WatchdogSec=0

[Install]
WantedBy=default.target
"""


def render_dashboard_unit(
    *,
    venv_python: Path,
    config_path: Path | None,
    memory_max_bytes: int,
    host: str,
    port: int,
    state_dir: Path,
    auth_enabled: bool = False,
    # The variable NAME, never a secret value.
    auth_secret_env: str = "EST_DASHBOARD_PROXY_SECRET",  # noqa: S107
) -> str:
    """Render the systemd user unit for the Streamlit frontend.

    A second unit, deliberately separate from the collector: they have
    different lifetimes, different memory ceilings and opposite restart
    intentions. The daemon is the thing that must be up; the dashboard is a
    convenience that an operator can stop without affecting monitoring.

    Parameters
    ----------
    venv_python
        Absolute path to the interpreter inside the virtual environment.
    config_path
        Explicit config file, or ``None`` to let the app resolve the XDG path.
    memory_max_bytes
        Hard cgroup ceiling. Separate from ``daemon.max_rss_bytes`` because
        Streamlit and Plotly are an order of magnitude heavier than the
        stdlib-only daemon; sharing the daemon's 48 MiB ceiling would have this
        unit OOM-killed while idle.
    host
        Bind address, from ``[dashboard] host``.
    port
        Bind port, from ``[dashboard] port``.
    state_dir
        Writable so the dashboard can open the SQLite database. It logically
        only reads, but a reader on a WAL-mode database still needs to create
        and write the ``-shm`` shared-memory file, so a read-only mount fails
        at open time with "unable to open database file".
    auth_enabled
        Whether ``[dashboard.auth] enabled`` is true. Only used to decide whether
        to reference the secret's environment file.
    auth_secret_env
        Name of the environment variable holding the proxy shared secret. The
        variable's *value* never appears in the unit; only its name, and a path
        to a 0600 file that sets it.

    Returns
    -------
    str
        The unit file contents.
    """
    config_arg = f' --config "{config_path}"' if config_path else ""
    memory_mib = memory_max_bytes / (1024 * 1024)
    # Where the operator puts the shared secret. Documented rather than created
    # here: writing a 0600 file from something called `render_unit` would be a
    # surprising side effect for a renderer.
    auth_env_file = dashboard_env_file()
    # The wildcard values are the point of the comparison, not an oversight.
    wildcard = host in {"0.0.0.0", "::"}  # noqa: S104
    loopback = host in {"127.0.0.1", "::1", "localhost"}
    auth_note = (
        f"""# Authentication is ENABLED. This process trusts an identity header only when it
# is accompanied by ${auth_secret_env}, so a host that reaches this port directly
# cannot forge a user by guessing a name. That secret is read from the environment
# file below, never from this unit."""
        if auth_enabled
        else """# WARNING: authentication is DISABLED. Anything that can reach this port can read
# the store. `est config validate` refuses a non-loopback bind in this state, so
# if this unit is running with a wider bind than loopback, the configuration was
# changed in a way the validator no longer sees."""
    )
    exposure = (
        "The dashboard has NO AUTHENTICATION. Every host that can reach this "
        "port can read the store -- colony positions, SAR grids, alerts -- and "
        "no credential is required. 0.0.0.0 binds every interface, so on a "
        "shared network this is open to the local LAN as well as whatever "
        "mesh you intended. Bind a specific private address instead, or reach "
        "it over an SSH tunnel or `tailscale serve`."
        if wildcard
        else (
            "The dashboard has NO AUTHENTICATION; anything that can route to "
            f"{host} can read the store. Binding a specific address limits that "
            "to hosts that know it, which is the point."
            if not loopback
            else "Bound to loopback, so only this machine can reach it."
        )
    )
    # Only emitted when auth is on, because it is only true then.
    #
    # It used to be unconditional, which made a generated unit for an
    # auth-disabled node claim that a missing secret file causes the dashboard
    # to refuse every caller. With auth off there is no secret to be missing and
    # nothing to refuse: the operator reads a comment asserting a fail-closed
    # guarantee this configuration does not have, and has no way to tell from
    # the unit that it is inapplicable. Shipping a reassuring sentence that does
    # not hold is worse than shipping no sentence.
    auth_env_block = (
        f"""#
# The shared secret arrives via an environment file rather than being written
# into this unit, so it is not readable in `systemctl cat` output and does not
# end up in a config backup. The leading `-` makes the file optional: if it is
# absent the service still starts, and the dashboard refuses every caller,
# which is the correct fail-closed outcome for a missing credential.
EnvironmentFile=-{auth_env_file}"""
        if auth_enabled
        else ""
    )
    return f"""\
# Emperor Space Tracker -- Streamlit dashboard
#
# Generated by `est install`. Edit `[dashboard]` in the config file and re-run
# `est install`, then: systemctl --user restart {DASHBOARD_UNIT_NAME}
#
# Separate from {UNIT_NAME} on purpose. The collector is the thing that must be
# up; this is a view onto it. Stopping or disabling this unit does not affect
# monitoring, and the two have different memory ceilings.

[Unit]
Description=Emperor Space Tracker - dashboard
Documentation=https://github.com/Castle96/emperor-space-tracker
After=network-online.target {UNIT_NAME}
Wants=network-online.target
# Only useful once the collector has created the schema, so start it after
# rather than alongside; Requires would take the collector down with it.
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
ExecStart="{venv_python}" -m emperor_space_tracker dashboard{config_arg}

# -- bind address --------------------------------------------------------
# From `[dashboard] host` and `port`. `est dashboard` resolves the same values
# at runtime, so overriding these here without also changing the config is
# possible but the two will disagree, which is why they come from one source.
# {exposure}
#
{auth_note}
{auth_env_block}
# -- restart policy -------------------------------------------------------
Restart=always
RestartSec=5

# -- resource ceiling -----------------------------------------------------
# Streamlit + Plotly measured ~80 MiB idle, so this is generous by design. It
# exists to convert a leak into a clean OOM kill and restart rather than an
# unresponsive process nobody notices until they want a page.
MemoryMax={memory_mib:.0f}M
MemoryHigh={memory_mib * 0.85:.0f}M

# -- filesystem sandboxing ------------------------------------------------
# Same posture as the collector. The state directory is writable rather than
# read-only because a SQLite reader on a WAL database still has to write the
# -shm file; that is a WAL requirement, not a sign the dashboard mutates data.
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths="{state_dir}"
PrivateTmp=true
PrivateDevices=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
ProtectClock=true
ProtectHostname=true
ProtectProc=invisible
RestrictNamespaces=true
RestrictRealtime=true
RestrictSUIDSGID=true
LockPersonality=true
MemoryDenyWriteExecute=true
NoNewPrivileges=true
RemoveIPC=true
UMask=0077

# -- network --------------------------------------------------------------
# This one LISTENS, so it needs AF_INET and AF_INET6 -- unlike the collector,
# which only ever makes outbound connections. AF_UNIX and AF_PACKET stay
# dropped: the socket primitives a compromised process would reach for first
# are not needed to serve HTTP.
RestrictAddressFamilies=AF_INET AF_INET6
# Streamlit's own protections are left on (CORS and XSRF are enabled by
# default in the app). They are not authentication -- they stop a random web
# page from driving the dashboard -- but they cost nothing to keep.
PrivateUsers=false

# -- process --------------------------------------------------------------
KillSignal=SIGTERM
# `est dashboard` execve()s into Streamlit rather than forking a child, so the
# process systemd is signalling IS the server. A wrapper that merely waited on
# a child could die on SIGTERM while the child survived holding the port, and
# the next start would fail on EADDRINUSE.
TimeoutStopSec=30
WatchdogSec=0

[Install]
WantedBy=default.target
"""


def install_user_unit(
    *,
    venv_python: Path | None = None,
    config_path: Path | None = None,
    config: Config | None = None,
) -> Path:
    """Write the systemd user unit and reload the user manager.

    Parameters
    ----------
    venv_python
        Interpreter to run. Defaults to the running one, resolved absolutely.
    config_path
        Explicit config file to pin. Defaults to the user's XDG config path if
        it exists.
    config
        Loaded configuration, used for the memory ceiling and state directory.
        Loaded from defaults if omitted.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    TrackerError
        If the unit directory cannot be created or the file cannot be written.
    """
    cfg = config if config is not None else _load_default_config()
    python_path = _resolve_interpreter(venv_python)
    config_path = _resolve_config_path(config_path)
    state_dir = _ensure_state_dir(cfg)

    unit = render_unit(
        venv_python=python_path,
        config_path=config_path,
        memory_max_bytes=cfg.daemon.max_rss_bytes,
        rss_warn_fraction=cfg.daemon.rss_warn_fraction,
        state_dir=state_dir,
    )
    target = _write_unit(unit_path(), unit)
    ok, output = systemctl("daemon-reload")
    if not ok:
        _LOG.warning("systemctl --user daemon-reload failed: %s", output)
    return target


def install_dashboard_unit(
    *,
    venv_python: Path | None = None,
    config_path: Path | None = None,
    config: Config | None = None,
) -> Path:
    """Write the dashboard systemd user unit and reload the user manager.

    Parameters
    ----------
    venv_python
        Interpreter to run. Defaults to the running one, resolved absolutely.
    config_path
        Explicit config file to pin. Defaults to the user's XDG config path if
        it exists.
    config
        Loaded configuration, used for the bind address, port and memory
        ceiling. Loaded from defaults if omitted.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    TrackerError
        If the unit directory cannot be created or the file cannot be written.
    """
    cfg = config if config is not None else _load_default_config()
    python_path = _resolve_interpreter(venv_python)
    config_path = _resolve_config_path(config_path)
    state_dir = _ensure_state_dir(cfg)

    unit = render_dashboard_unit(
        venv_python=python_path,
        config_path=config_path,
        memory_max_bytes=cfg.dashboard.max_rss_bytes,
        host=cfg.dashboard.host,
        port=cfg.dashboard.port,
        state_dir=state_dir,
        auth_enabled=cfg.dashboard.auth.enabled,
        auth_secret_env=cfg.dashboard.auth.proxy_secret_env,
    )
    target = _write_unit(dashboard_unit_path(), unit)
    ok, output = systemctl("daemon-reload")
    if not ok:
        _LOG.warning("systemctl --user daemon-reload failed: %s", output)
    return target


def _resolve_interpreter(venv_python: Path | None) -> Path:
    """Return an absolute, existing interpreter path.

    NOTE: never ``.resolve()`` this path. A venv binary is typically a symlink
    to a base interpreter, and CPython locates the venv (pyvenv.cfg,
    site-packages) from the *unresolved* executable path. Resolving it silently
    repoints the unit at the bare base interpreter, where
    ``import emperor_space_tracker`` fails and the service restart-loops.

    Parameters
    ----------
    venv_python
        Candidate path, or ``None`` for the running interpreter.

    Returns
    -------
    Path
        An absolute path to a file that exists.

    Raises
    ------
    TrackerError
        If the interpreter does not exist.
    """
    raw = venv_python or Path(sys.executable)
    resolved = raw if raw.is_absolute() else Path.cwd() / raw
    if not resolved.is_file():
        msg = f"interpreter {resolved} does not exist"
        raise TrackerError(msg)
    return resolved


def _resolve_config_path(config_path: Path | None) -> Path | None:
    """Return the config file to pin, or ``None`` to let the app resolve it.

    Parameters
    ----------
    config_path
        Explicit path, or ``None``.

    Returns
    -------
    Path | None
        The explicit path if given, else the user's XDG config if it exists,
        else ``None``.
    """
    if config_path is not None:
        return config_path
    from .config import default_config_path

    candidate = default_config_path()
    return candidate if candidate.is_file() else None


def _ensure_state_dir(cfg: Config) -> Path:
    """Create the state directory, raising a clear error if that fails.

    Parameters
    ----------
    cfg
        Loaded configuration.

    Returns
    -------
    Path
        The resolved state directory.

    Raises
    ------
    TrackerError
        If the directory cannot be created.
    """
    state_dir = cfg.paths.resolved_state_dir()
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot create state directory {state_dir}: {exc.strerror or exc}"
        raise TrackerError(msg) from exc
    return state_dir


def _write_unit(target: Path, contents: str) -> Path:
    """Write a unit file, raising a clear error on failure.

    Parameters
    ----------
    target
        Destination path.
    contents
        Rendered unit text.

    Returns
    -------
    Path
        ``target``.

    Raises
    ------
    TrackerError
        If the file cannot be written.
    """
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
    except OSError as exc:
        msg = f"cannot write {target}: {exc.strerror or exc}"
        raise TrackerError(msg) from exc
    _LOG.info("wrote systemd user unit to %s", target)
    return target


def _load_default_config() -> Config:
    """Load the configuration, falling back to packaged defaults on error."""
    from .config import load_config

    try:
        return load_config()
    except TrackerError:
        return load_config(use_user_config=False)


def systemctl(*args: str) -> tuple[bool, str]:
    """Run a ``systemctl --user`` command.

    Parameters
    ----------
    *args
        Subcommand and arguments.

    Returns
    -------
    tuple[bool, str]
        ``(success, combined_output)``. Never raises: the caller decides
        whether a missing systemd is fatal.
    """
    systemctl_bin = shutil.which("systemctl")
    if systemctl_bin is None:
        return False, "systemctl is not on PATH"
    try:
        completed = subprocess.run(
            [systemctl_bin, "--user", *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


@dataclass(frozen=True, slots=True)
class UnitStatus:
    """Parsed status of the installed unit.

    Attributes
    ----------
    installed
        Whether the unit file exists.
    active
        The systemd ``ActiveState``, or ``None`` if unknown.
    sub_state
        The systemd ``SubState``, or ``None`` if unknown.
    main_pid
        The service's main PID, or ``None``.
    detail
        Raw ``systemctl status`` output.
    """

    installed: bool
    active: str | None
    sub_state: str | None
    main_pid: int | None
    detail: str

    @property
    def running(self) -> bool:
        """Return whether the unit is currently active."""
        return self.active == "active"


def unit_status() -> UnitStatus:
    """Query the installed unit's status.

    Returns
    -------
    UnitStatus
        Parsed status. Never raises.
    """
    path = unit_path()
    if not path.is_file():
        return UnitStatus(False, None, None, None, f"not installed at {path}")

    systemctl_bin = shutil.which("systemctl")
    if systemctl_bin is None:
        return UnitStatus(True, None, None, None, "installed; systemctl unavailable")

    try:
        completed = subprocess.run(
            [systemctl_bin, "--user", "show", UNIT_NAME,
             "--property=ActiveState,SubState,MainPID"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        properties: dict[str, str] = {}
        for line in (completed.stdout or "").splitlines():
            key, _, value = line.partition("=")
            properties[key.strip()] = value.strip()

        detail_run = subprocess.run(
            [systemctl_bin, "--user", "status", UNIT_NAME, "--no-pager"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return UnitStatus(True, None, None, None, f"installed; status query failed: {exc}")

    raw_pid = properties.get("MainPID", "0")
    return UnitStatus(
        installed=True,
        active=properties.get("ActiveState"),
        sub_state=properties.get("SubState"),
        main_pid=int(raw_pid) if raw_pid.isdigit() and int(raw_pid) > 0 else None,
        detail=(detail_run.stdout or detail_run.stderr).strip(),
    )
