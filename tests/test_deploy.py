"""Tests for the generated systemd user unit.

The unit file is the one artifact that cannot be exercised by importing the
package, so these tests assert on the rendered text. Each assertion corresponds
to a property that is easy to break by editing the template and hard to notice
until a field node is misbehaving.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from emperor_space_tracker.config import load_config
from emperor_space_tracker.deploy import (
    DASHBOARD_UNIT_NAME,
    UNIT_NAME,
    render_dashboard_unit,
    render_unit,
)


@pytest.fixture
def unit() -> str:
    """Render a unit with plausible values for every required argument."""
    return render_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=48 * 1024 * 1024,
        rss_warn_fraction=0.85,
        state_dir=Path("/home/node/.local/state/emperor-space-tracker"),
    )


def _directives(text: str) -> dict[str, list[str]]:
    """Parse a rendered unit into a mapping of key to every value seen.

    Repeated keys are kept in order, because a duplicated directive is itself a
    bug: systemd honours the last one, so a stray duplicate silently overrides
    the line above it.
    """
    out: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out.setdefault(key.strip(), []).append(value.strip())
    return out


def test_warn_threshold_sits_above_the_measured_peak() -> None:
    """The default warning threshold must clear the observed poll peak.

    A full polling pass measures ~35.8-36.2 MiB. At the old 0.75 fraction the
    threshold landed on 36.0 MiB, which that peak straddles, so the daemon
    logged a loud RSS warning on essentially every single pass. That is worse
    than no warning at all: it trains an operator to ignore the one that would
    matter. The threshold has to sit above the peak, not merely below the
    ceiling.
    """
    from emperor_space_tracker.config import load_config

    config = load_config(use_user_config=False)
    daemon = config.daemon
    peak_mib = 36.2
    threshold_mib = config.effective_rss_mib() * daemon.rss_warn_fraction

    assert threshold_mib > peak_mib, (
        f"warning threshold {threshold_mib:.1f} MiB is not above the measured "
        f"peak {peak_mib} MiB, so the warning fires on every pass"
    )
    # Still comfortably below the hard limit, so drift is caught before the
    # cgroup turns it into an OOM kill and restart loop.
    assert threshold_mib < config.effective_rss_mib()


def test_unit_quotes_the_configured_warn_fraction() -> None:
    """The rendered comment states the real threshold, not a stale constant."""
    rendered = render_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=48 * 1024 * 1024,
        rss_warn_fraction=0.85,
        state_dir=Path("/home/node/.local/state/emperor-space-tracker"),
    )
    assert "41 MiB" in rendered  # 48 * 0.85 = 40.8
    assert "36 MiB" not in rendered


def test_unit_renders_the_expected_interpreter_and_command(unit: str) -> None:
    """ExecStart invokes the module inside the venv, not a bare ``python``."""
    assert 'ExecStart="/opt/venv/bin/python" -m emperor_space_tracker run' in unit


def test_install_keeps_the_venv_symlink_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unit must reference the venv binary, not the resolved interpreter.

    A venv ``bin/python`` is a symlink to a base interpreter, and CPython finds
    the venv from the unresolved path. Resolving it repointed ExecStart at the
    bare base interpreter, where the package is not installed, and the service
    restart-looped with ``No module named emperor_space_tracker``.
    """
    import sys

    import emperor_space_tracker.deploy as deploy_mod
    from emperor_space_tracker.deploy import install_user_unit

    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(Path(sys.executable).resolve())
    target = tmp_path / "emperor-space-tracker.service"
    monkeypatch.setattr(deploy_mod, "unit_path", lambda: target)
    monkeypatch.setattr(deploy_mod, "systemctl", lambda *args: (True, ""))

    install_user_unit(venv_python=link)
    unit = target.read_text()
    assert f'ExecStart="{link}" -m emperor_space_tracker run' in unit
    assert str(link.resolve()) not in unit or str(link) == str(link.resolve())


def test_memory_max_matches_the_configured_ceiling(unit: str) -> None:
    """The cgroup limit tracks ``daemon.max_rss_bytes`` rather than a constant."""
    directives = _directives(unit)
    assert directives["MemoryMax"] == ["48M"]
    # MemoryHigh must stay below the hard limit or the soft threshold is moot.
    assert int(directives["MemoryHigh"][0].rstrip("M")) < 48


def test_state_directory_is_the_only_writable_path(unit: str) -> None:
    """``ProtectSystem=strict`` with a single ``ReadWritePaths`` entry."""
    directives = _directives(unit)
    assert directives["ProtectSystem"] == ["strict"]
    assert directives["ReadWritePaths"] == [
        '"/home/node/.local/state/emperor-space-tracker"'
    ]


def test_watchdog_is_disabled(unit: str) -> None:
    """A non-zero WatchdogSec would kill a healthy, non-notifying daemon."""
    assert _directives(unit)["WatchdogSec"] == ["0"]


def test_no_directive_is_duplicated(unit: str) -> None:
    """Systemd honours the last of a repeated key, so duplicates hide bugs.

    ``IPAddressAllow`` in particular appeared twice here, the second of which
    (``0.0.0.0/0``) undid the first while still reading like a restriction.
    """
    directives = _directives(unit)
    duplicated = {k: v for k, v in directives.items() if len(v) > 1}
    assert duplicated == {}


def test_address_families_exclude_local_and_raw_sockets(unit: str) -> None:
    """Only the families needed to make outbound HTTP requests are available."""
    families = set(_directives(unit)["RestrictAddressFamilies"][0].split())
    assert families == {"AF_INET", "AF_INET6"}
    assert "AF_UNIX" not in families
    assert "AF_PACKET" not in families


def test_unit_is_a_user_unit_with_no_privilege_escalation(unit: str) -> None:
    """The service runs unprivileged, which is the point of a user unit."""
    directives = _directives(unit)
    assert "User=" not in directives  # inherits the invoking account
    assert directives["NoNewPrivileges"] == ["true"]
    assert "[Install]" in unit
    assert directives["WantedBy"] == ["default.target"]


def test_restart_policy_survives_a_clean_exit(unit: str) -> None:
    """``always``, not ``on-failure``: an unattended node must not die quietly."""
    assert _directives(unit)["Restart"] == ["always"]


def test_rendered_unit_passes_systemd_analyze_verify(tmp_path: Path, unit: str) -> None:
    """Confirm that systemd itself accepts the rendered file.

    Skipped where systemd is unavailable, which is the norm inside containers.
    A template that no longer parses still imports and still passes the tests
    above, so this is the only check that would catch a syntax regression.
    """
    binary = shutil.which("systemd-analyze")
    if binary is None:
        pytest.skip("systemd-analyze is not available")

    path = tmp_path / "emperor-space-tracker.service"
    path.write_text(unit)
    # Point ExecStart at an interpreter that exists so verify does not fail on
    # a missing binary, which is unrelated to the unit's syntax.
    path.write_text(unit.replace("/opt/venv/bin/python", "/usr/bin/python3"))

    result = subprocess.run(
        [binary, "verify", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# The dashboard unit
# --------------------------------------------------------------------------- #


@pytest.fixture
def dash_unit() -> str:
    """Render a dashboard unit with plausible values."""
    return render_dashboard_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=512 * 1024 * 1024,
        host="127.0.0.1",
        port=8501,
        state_dir=Path("/home/node/.local/state/emperor-space-tracker"),
    )


def test_the_dashboard_unit_is_separate_from_the_collector() -> None:
    """Two units, two names, so either can be stopped independently.

    The collector is the thing that must be up; the dashboard is a view onto it.
    Sharing a unit would mean a hung web server takes the monitoring with it.
    """
    assert "dashboard" in DASHBOARD_UNIT_NAME
    assert "dashboard" not in UNIT_NAME
    assert render_dashboard_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=512 * 1024 * 1024,
        host="127.0.0.1",
        port=8501,
        state_dir=Path("/s"),
    ).count("[Service]") == 1


def test_the_dashboard_unit_launches_est_dashboard(dash_unit: str) -> None:
    """It runs `est dashboard`, not streamlit directly.

    Going through the CLI is what makes the unit's ExecStart and an operator's
    manual `est dashboard` resolve the same bind address from the same config,
    instead of two code paths that can disagree.
    """
    assert "emperor_space_tracker dashboard" in dash_unit
    assert "streamlit run" not in dash_unit


def test_the_dashboard_ceiling_is_independent_of_the_daemon(dash_unit: str) -> None:
    """Streamlit needs its own ceiling, far above the daemon's 48 MiB.

    Applying the daemon's budget here would have the unit OOM-killed while
    merely idle, since Streamlit plus Plotly measure ~80 MiB.
    """
    assert "MemoryMax=512M" in dash_unit
    assert "MemoryHigh=435M" in dash_unit


def test_the_dashboard_unit_can_listen(dash_unit: str) -> None:
    """It needs AF_INET/AF_INET6 to bind, unlike the collector."""
    directive = next(
        line
        for line in dash_unit.splitlines()
        if line.startswith("RestrictAddressFamilies=")
    )
    assert directive == "RestrictAddressFamilies=AF_INET AF_INET6"
    # AF_UNIX and AF_PACKET stay dropped even for a server. Checked against the
    # directive rather than the file: the header comment names them precisely to
    # record that they were considered and excluded.
    assert "AF_UNIX" not in directive
    assert "AF_PACKET" not in directive


def test_the_dashboard_state_dir_is_writable_for_wal(dash_unit: str) -> None:
    """A WAL reader still has to write the -shm file.

    `ReadOnlyPaths` on the state directory would fail the dashboard at open
    time with "unable to open database file" even though it never writes, which
    is a genuinely confusing error for a read-only workload.
    """
    assert 'ReadWritePaths="/home/node/.local/state/emperor-space-tracker"' in dash_unit
    assert "ReadOnlyPaths" not in dash_unit


def test_a_loopback_bind_says_so(dash_unit: str) -> None:
    """The unit documents its actual exposure rather than a generic warning."""
    assert "Bound to loopback" in dash_unit


def test_a_disabled_auth_dashboard_unit_warns_loudly() -> None:
    """An unauthenticated dashboard must say so in the unit itself.

    The unit is the artifact an operator reads months later, when they have
    forgotten the dashboard was ever unauthenticated.
    """
    unit_text = render_dashboard_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=512 * 1024 * 1024,
        host="127.0.0.1",
        port=8501,
        state_dir=Path("/s"),
        auth_enabled=False,
    )
    assert "authentication is DISABLED" in unit_text


def test_a_disabled_auth_unit_makes_no_claim_about_a_missing_secret() -> None:
    """An auth-off unit must not assert a fail-closed guarantee it lacks.

    This one shipped the "if the file is absent the service still starts, and
    the dashboard refuses every caller" paragraph unconditionally. With auth off
    there is no secret to be missing and nothing to refuse, so an operator
    reading the generated unit was told their unauthenticated dashboard fails
    closed. It does not. Shipping a reassuring sentence that does not hold is
    worse than shipping no sentence.
    """
    unit_text = render_dashboard_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=512 * 1024 * 1024,
        host="127.0.0.1",
        port=8501,
        state_dir=Path("/s"),
        auth_enabled=False,
    )
    assert "EnvironmentFile" not in unit_text
    assert "fail-closed" not in unit_text
    assert "refuses every caller" not in unit_text


def test_the_auth_secret_paragraph_appears_only_when_auth_is_on() -> None:
    """Both halves of the invariant, so the fix cannot be undone one-sidedly.

    Dropping the paragraph while auth is on would remove the only instruction
    telling an operator where the secret has to live, and the service would
    start with no credential and refuse everyone with no explanation.
    """
    def render(auth_enabled: bool) -> str:
        return render_dashboard_unit(
            venv_python=Path("/opt/venv/bin/python"),
            config_path=None,
            memory_max_bytes=512 * 1024 * 1024,
            host="127.0.0.1",
            port=8501,
            state_dir=Path("/s"),
            auth_enabled=auth_enabled,
            auth_secret_env="MY_SECRET_VAR",
        )

    assert "EnvironmentFile=-" in render(True)
    assert "EnvironmentFile=-" not in render(False)


def test_an_enabled_auth_dashboard_unit_names_the_secret() -> None:
    """The unit points at the environment file, never at the secret's value.

    A secret written into a unit file is readable via `systemctl cat` by every
    user session on the machine and turns up in config backups.
    """
    unit_text = render_dashboard_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=512 * 1024 * 1024,
        host="127.0.0.1",
        port=8501,
        state_dir=Path("/s"),
        auth_enabled=True,
        auth_secret_env="MY_SECRET_VAR",
    )
    assert "Authentication is ENABLED" in unit_text
    assert "EnvironmentFile=-" in unit_text
    assert "MY_SECRET_VAR" in unit_text
    assert "dashboard.env" in unit_text


def test_the_environment_file_is_optional_so_a_missing_secret_still_starts() -> None:
    """The leading `-` is what makes the unit start without the file.

    Without it, a missing secret file is a unit that will not start, so the
    operator sees a dead service rather than a dashboard that refuses everyone.
    Both are safe; only one is diagnosable.
    """
    unit_text = render_dashboard_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=512 * 1024 * 1024,
        host="127.0.0.1",
        port=8501,
        state_dir=Path("/s"),
        auth_enabled=True,
    )
    line = next(
        row for row in unit_text.splitlines() if row.startswith("EnvironmentFile=")
    )
    assert line.startswith("EnvironmentFile=-")


def test_a_wildcard_bind_warns_about_the_missing_authentication() -> None:
    """Binding 0.0.0.0 has to say what it exposes, in the unit itself.

    The unit is the artifact an operator reads months later, and it is the only
    place guaranteed to be looked at before a node is put on a shared network.
    """
    unit_text = render_dashboard_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=512 * 1024 * 1024,
        # A wildcard bind is what this test is about.
        host="0.0.0.0",  # noqa: S104
        port=8501,
        state_dir=Path("/s"),
    )
    assert "NO AUTHENTICATION" in unit_text
    assert "0.0.0.0 binds every interface" in unit_text
    # A wildcard bind is not loopback, so it must not claim to be.
    assert "Bound to loopback" not in unit_text


def test_a_specific_private_bind_is_described_accurately() -> None:
    """A Tailscale address gets its own wording, not the wildcard warning."""
    unit_text = render_dashboard_unit(
        venv_python=Path("/opt/venv/bin/python"),
        config_path=None,
        memory_max_bytes=512 * 1024 * 1024,
        host="100.112.18.30",
        port=8501,
        state_dir=Path("/s"),
    )
    assert "100.112.18.30" in unit_text
    assert "0.0.0.0 binds every interface" not in unit_text
    assert "NO AUTHENTICATION" in unit_text


def test_the_dashboard_restarts_on_a_clean_exit(dash_unit: str) -> None:
    """An unexpected clean exit should come back, as for the collector."""
    assert "Restart=always" in dash_unit
    assert "RestartSec=" in dash_unit


def test_the_dashboard_depends_on_nothing_it_can_break(dash_unit: str) -> None:
    """`After=` the collector, but never `Requires=` it.

    `Requires` would mean a collector restart takes the dashboard down with it,
    and stopping the collector would stop the dashboard -- coupling a
    convenience to the thing that actually matters.
    """
    assert f"After=network-online.target {UNIT_NAME}" in dash_unit
    assert "Requires=" not in dash_unit
    assert "BindsTo=" not in dash_unit


def test_the_dashboard_documents_why_it_execvees(dash_unit: str) -> None:
    """The exec-not-fork decision is recorded where an operator will read it.

    A wrapper that waits on a child can die on SIGTERM while the child survives
    holding the port, and the next start then fails on EADDRINUSE forever.
    """
    assert "execve" in dash_unit
    assert "EADDRINUSE" in dash_unit


def test_the_dashboard_has_no_duplicate_directives(dash_unit: str) -> None:
    """Repeated keys are a syntax error, not a warning, in a unit file."""
    directives = [
        line.split("=", 1)[0]
        for line in dash_unit.splitlines()
        if line and not line.startswith(("#", " ", "\t")) and "=" in line
    ]
    duplicates = {d for d in directives if directives.count(d) > 1}
    assert not duplicates, f"duplicated directives: {duplicates}"


def test_the_dashboard_unit_passes_systemd_analyze_verify(
    tmp_path: Path, dash_unit: str
) -> None:
    """Confirm that systemd itself accepts the rendered dashboard file."""
    binary = shutil.which("systemd-analyze")
    if binary is None:
        pytest.skip("systemd-analyze is not available")

    path = tmp_path / "emperor-space-tracker-dashboard.service"
    path.write_text(dash_unit.replace("/opt/venv/bin/python", "/usr/bin/python3"))

    result = subprocess.run(
        [binary, "verify", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_installing_both_units_is_idempotent(tmp_path: Path) -> None:
    """Re-running the installers overwrites rather than duplicating.

    `est install` is documented as re-runnable after a config change, so a
    second run must produce the same file and not append to it.
    """
    from emperor_space_tracker.deploy import dashboard_unit_path, install_dashboard_unit, unit_path

    config = load_config(use_user_config=False)
    python = Path(sys.executable)

    first = install_dashboard_unit(venv_python=python, config_path=None, config=config)
    second = install_dashboard_unit(venv_python=python, config_path=None, config=config)
    assert first == second
    assert first.read_text() == second.read_text()
    assert unit_path().name == UNIT_NAME
    assert dashboard_unit_path() != unit_path()
