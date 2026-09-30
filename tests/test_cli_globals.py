"""Regression tests for the global-flag parsing in :mod:`emperor_space_tracker.cli`.

Every one of these cases was a real defect at some point, and all of them fail
silently: the command still exits 0, it just quietly ignores what was asked for.
That is why they are pinned by tests rather than left to a usage note.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from emperor_space_tracker.cli import _apply_global_defaults, build_parser

_FLAG_CASES = [
    pytest.param(["-v", "poll"], "verbose", True, id="verbose-before"),
    pytest.param(["poll", "-v"], "verbose", True, id="verbose-after"),
    pytest.param(["-q", "poll"], "quiet", True, id="quiet-before"),
    pytest.param(["poll", "-q"], "quiet", True, id="quiet-after"),
    pytest.param(["--no-user-config", "status"], "no_user_config", True,
                 id="no-user-config-before"),
    pytest.param(["status", "--no-user-config"], "no_user_config", True,
                 id="no-user-config-after"),
    pytest.param(["-c", "config.toml", "poll"], "config", "config.toml",
                 id="config-before"),
    pytest.param(["poll", "-c", "config.toml"], "config", "config.toml",
                 id="config-after"),
]


@pytest.mark.parametrize(("argv", "attribute", "expected"), _FLAG_CASES)
def test_shared_flags_parse_on_either_side_of_the_command(
    argv: list[str], attribute: str, expected: object
) -> None:
    """A shared flag means the same thing before or after the subcommand."""
    args = build_parser().parse_args(argv)
    _apply_global_defaults(args)
    assert getattr(args, attribute) == expected


@pytest.mark.parametrize(("flag", "attribute"), [("-v", "verbose"), ("--verbose", "verbose"),
                                                   ("-q", "quiet"), ("--quiet", "quiet")])
def test_a_flag_given_before_the_command_is_not_overwritten(
    flag: str, attribute: str
) -> None:
    """The subparser's default must not clobber a value set at the top level.

    This is the defect that made ``est -v poll`` a no-op. argparse parses the
    top-level flags into the namespace, then the subparser applies its own
    defaults, so any shared flag carrying a real default is reset unless the
    parser suppresses defaults and they are reapplied afterwards.
    """
    before = build_parser().parse_args([flag, "poll"])
    _apply_global_defaults(before)
    assert getattr(before, attribute) is True


def test_omitted_shared_flags_take_their_defaults() -> None:
    """Unset shared flags resolve to the documented defaults."""
    args = build_parser().parse_args(["poll"])
    _apply_global_defaults(args)
    assert args.config is None
    assert args.verbose is False
    assert args.quiet is False
    assert args.no_user_config is False


def test_no_command_parses_without_error() -> None:
    """A bare invocation still parses, leaving the caller to print help."""
    args = build_parser().parse_args([])
    _apply_global_defaults(args)
    assert getattr(args, "command", None) is None


def test_unknown_command_is_rejected() -> None:
    """A misspelled subcommand is an error, not a silent no-op."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nonsense"])


def test_repeated_flag_keeps_the_last_value() -> None:
    """``-c a -c b`` means b, so later occurrences win in both positions."""
    args = build_parser().parse_args(["-c", "a.toml", "poll", "-c", "b.toml"])
    _apply_global_defaults(args)
    assert args.config == "b.toml"


def test_namespace_helper_is_idempotent() -> None:
    """Reapplying defaults does not clobber an explicitly set value."""
    args = build_parser().parse_args(["-v", "poll"])
    _apply_global_defaults(args)
    _apply_global_defaults(args)
    assert isinstance(args, argparse.Namespace)
    assert args.verbose is True


# --------------------------------------------------------------------------- #
# dashboard flag / config resolution
# --------------------------------------------------------------------------- #


def test_dashboard_flags_default_to_none_so_config_wins() -> None:
    """Every dashboard flag must be optional, or config can never take effect.

    A flag with a built-in default always wins over `[dashboard]`, which is how
    a node deliberately bound to a tailnet address silently reverts to loopback
    the first time someone runs `est dashboard` without repeating the flags.
    """
    args = build_parser().parse_args(["dashboard"])
    assert args.host is None
    assert args.port is None
    assert args.headless is None


def test_dashboard_flags_still_override() -> None:
    """An explicit flag beats the config, as CLI options always must."""
    # A wildcard bind is the point: a configured tailnet address must stay
    # overridable by an explicit flag.
    wildcard = "0.0.0.0"  # noqa: S104 -- the point of the assertion
    args = build_parser().parse_args(
        ["dashboard", "--host", wildcard, "--port", "9999", "--open"]
    )
    assert args.host == wildcard
    assert args.port == 9999
    assert args.headless is False


def test_dashboard_headless_and_open_are_mutually_exclusive() -> None:
    """Asking for both is a contradiction, not a silent precedence rule."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["dashboard", "--headless", "--open"])


def test_global_flags_still_work_either_side_of_dashboard(tmp_path: Path) -> None:
    """`-c` must reach the subcommand whichever side it is written on."""
    config = str(tmp_path / "x.toml")
    for argv in (
        ["-c", config, "dashboard"],
        ["dashboard", "-c", config],
    ):
        args = build_parser().parse_args(argv)
        _apply_global_defaults(args)
        assert args.config == config


def test_uninstall_is_registered() -> None:
    """Teardown exists, so a unit can be removed without editing files by hand."""
    args = build_parser().parse_args(["uninstall"])
    assert callable(args.func)
    assert args.func.__name__ == "_cmd_uninstall"


def test_install_can_skip_the_dashboard_unit() -> None:
    """`--no-dashboard` for a node that should expose nothing."""
    args = build_parser().parse_args(["install", "--no-dashboard"])
    assert args.no_dashboard is True
    assert build_parser().parse_args(["install"]).no_dashboard is False
