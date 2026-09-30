"""Smoke tests that actually execute the Streamlit app.

An HTTP 200 from the Streamlit server proves nothing about this module: Streamlit
serves its shell on every request and only runs the script over a websocket once
a browser connects. These tests use Streamlit's own ``AppTest`` harness, which
executes the script the way the server does and surfaces any exception raised,
so a dashboard that renders a blank page or crashes on rerun fails here.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from emperor_space_tracker.store import _SCHEMA

APP = (
    Path(__file__).resolve().parents[1]
    / "src" / "emperor_space_tracker" / "dashboard" / "app.py"
)


@pytest.fixture
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the dashboard at an empty throwaway database.

    The dashboard resolves its own store from the config, so without this the
    test would read (and potentially prune) the operator's real database.
    """
    database = tmp_path / "tracker.sqlite3"
    conn = sqlite3.connect(database)
    conn.executescript(_SCHEMA)
    conn.close()

    config = tmp_path / "config.toml"
    config.write_text(f'[paths]\nstate_dir = "{tmp_path}"\n')
    monkeypatch.setenv("EST_CONFIG", str(config))
    monkeypatch.setenv("EST_NO_USER_CONFIG", "1")
    return database


def test_dashboard_runs_against_an_empty_store(isolated_store: Path) -> None:
    """Every tab renders with no data at all, which is the first-run state.

    An empty database is the state a fresh node is in, and it is where
    None-handling bugs surface: a query that assumes a row exists would render
    fine against a populated store and fail on a real first run.
    """
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert [t.label for t in at.tabs] == [
        "overview", "space weather", "fast ice", "colonies", "alerts",
    ]


def test_dashboard_window_selector_reruns_cleanly(isolated_store: Path) -> None:
    """Changing the sidebar window re-queries without raising.

    Streamlit reruns the whole script on every widget change, so the store
    connection is opened and closed once per rerun. A connection leaked or
    closed at the wrong point shows up here and nowhere else.
    """
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    for label in ("6 hours", "7 days", "30 days", "24 hours"):
        at.sidebar.selectbox[0].select(label).run()
        assert not at.exception, f"{label}: {[str(e.value) for e in at.exception]}"


def test_dashboard_imports_without_the_extra_installed() -> None:
    """The module must not import Streamlit at module scope.

    A daemon node has no dashboard extra, and ``est doctor`` and ``--help`` have
    to keep working there. A top-level ``import streamlit`` would make the
    whole package unimportable without the extra.
    """
    tree = ast.parse(APP.read_text())
    module_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_level.add(node.module or "")
            module_level.update(a.name for a in node.names)

    assert "streamlit" not in module_level
    assert "plotly" not in module_level
    assert "earthengine" not in module_level


def test_dashboard_has_a_script_entry_point() -> None:
    """The file renders when Streamlit executes it as ``__main__``.

    Without this guard the module only defines functions, the server starts
    happily, and every visitor sees an empty page.
    """
    tree = ast.parse(APP.read_text())
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.dump(node.test).find("__main__") != -1
    ]
    assert guards, "no `if __name__ == '__main__':` entry point in the dashboard"


def test_dashboard_uses_absolute_imports() -> None:
    """Relative imports break when Streamlit runs the file as a script.

    Streamlit executes the target file directly rather than importing it as part
    of its package, so ``from ..store import Store`` raises "attempted relative
    import with no known parent package" and the page never renders.
    """
    tree = ast.parse(APP.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            pytest.fail(
                f"line {node.lineno}: relative import in a Streamlit entry script"
            )


def test_colony_map_renders_with_data() -> None:
    """The colonies-tab map builds a valid Plotly figure from real rows.

    The AppTest smoke tests only execute the default tab, so a bad projection
    type in ``_render_colony_map`` crashed the live colonies tab while every
    gate stayed green. Plotly validates ``layout.geo.projection`` eagerly, so
    rendering one colony against a stub Streamlit pins the projection string.
    """
    pytest.importorskip("plotly")
    from unittest.mock import MagicMock

    import plotly.graph_objects as go

    from emperor_space_tracker.dashboard.app import _render_colony_map

    colonies = [
        {
            "colony_id": "cpe-test",
            "name": "Test Colony",
            "region": "Ross Sea",
            "population_estimate": 6000,
            "population_year": 2020,
            "population_source": "test",
            "fast_ice_ratio": 0.9,
            "notes": "",
            "latitude": -77.85,
            "longitude": 166.67,
            "stability": "stable",
        }
    ]
    st = MagicMock()
    _render_colony_map(st, go, colonies)
    (figure,) = st.plotly_chart.call_args[0]
    assert figure.layout.geo.projection.type == "stereographic"
    assert figure.layout.geo.projection.rotation.lat == -90


def test_dashboard_closes_the_store_from_a_finally_block() -> None:
    """The connection is closed on every exit path, not just the happy one.

    Streamlit re-executes the script on every rerun, so a leaked WAL connection
    accumulates for the life of the process. The guarantee has to be structural:
    ``close()`` in a ``finally`` owned by the same function that opened the
    store. A trailing ``store.close()`` after the last render satisfies a
    text-position check while still leaking whenever a widget raises.
    """
    tree = ast.parse(APP.read_text())
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    main = functions["main"]
    opened = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Store"
        for node in ast.walk(main)
    )
    assert opened, "main() should be the function that opens the store"

    # Every path out of main() must pass through a close().
    closed_in_finally = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
        for handler in (n for n in ast.walk(main) if isinstance(n, ast.Try))
        for statement in handler.finalbody
        for node in ast.walk(statement)
    )
    assert closed_in_finally, "store.close() must live in a finally block, not at the end"


def test_no_store_read_outside_the_render_function() -> None:
    """Store reads are confined to the function that receives an open store.

    ``main`` opens, delegates to ``_render`` and closes. Any read in ``main``
    after the delegation would sit outside the ``try`` that owns the
    connection.
    """
    tree = ast.parse(APP.read_text())
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    for name in ("main", "_render"):
        assert name in functions, f"{name}() should exist"

    main = functions["main"]
    renders = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_render"
    ]
    assert len(renders) == 1, "main() should delegate to _render exactly once"

    # Inside main(), the only `store.` call allowed is close().
    calls = [
        node.func.attr
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "store"
    ]
    assert set(calls) <= {"close"}, f"main() performs store reads: {set(calls)}"


# --------------------------------------------------------------------------- #
# The SAR heatmap must actually have data to draw
# --------------------------------------------------------------------------- #


@pytest.fixture
def seeded_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build an isolated store holding one scene with a real 5x5 grid.

    The empty-store fixture proves the tabs survive first run. This one proves
    the populated path draws, which is where the heatmap was silently broken:
    an empty store cannot tell "no data yet" apart from "the lookup is wrong".
    """
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from datetime import UTC, datetime

    from emperor_space_tracker.models import FastIceCell, SarScene
    from emperor_space_tracker.store import Store

    database = tmp_path / "seeded.sqlite3"
    conn = sqlite3.connect(database)
    conn.executescript(_SCHEMA)
    conn.close()

    now = datetime.now(UTC)
    cells = tuple(
        FastIceCell(
            scene_id="S1",
            row=r,
            col=c,
            sigma0_db=(-20.0 if (r + c) % 7 == 0 else -6.0),
            is_open_water=(r + c) % 7 == 0,
            classification="open_water" if (r + c) % 7 == 0 else "consolidated_ice",
        )
        for r in range(5)
        for c in range(5)
    )
    store = Store(str(database))
    try:
        store.record_sar_scene(
            SarScene(
                scene_id="S1",
                observed_at=now,
                platform="SENTINEL-1",
                orbit_type="ASCENDING",
                polarisation="VV",
                incidence_angle_deg=35.0,
                colony_id="cpe-test",
                mean_db=-7.0,
                min_db=-20.0,
                max_db=-2.0,
                std_db=3.0,
                frozen_ice_fraction=0.8,
                open_water_fraction=0.2,
                cells=cells,
                provenance="test/v1",
            )
        )
    finally:
        store.close()

    config = tmp_path / "seeded-config.toml"
    config.write_text(f'[paths]\nstate_dir = "{tmp_path}"\n')
    monkeypatch.setenv("EST_CONFIG", str(config))
    monkeypatch.setenv("EST_NO_USER_CONFIG", "1")
    return database


def test_sar_cells_are_read_from_the_cells_table_not_the_scene_row(
    seeded_store: Path,
) -> None:
    """The heatmap is fed from ``sar_cells``, which is where the grid lives.

    ``sar_scenes`` has no ``cells`` column, so reading ``scene["cells"]`` gives
    ``None``, ``n`` computes as 0, and the renderer reports "no SAR grid
    available" -- for every scene ever recorded, while 1,681 cells per scene sat
    unread. This asserts both halves: the scene row genuinely has no cells, and
    the grid is recoverable from the table that does.
    """
    from emperor_space_tracker.dashboard.app import _sar_grid
    from emperor_space_tracker.store import Store

    store = Store(str(seeded_store))
    try:
        scene = store.recent_sar_scenes(limit=1)[0]
        assert "cells" not in scene, "the scene row must not be the grid source"
        assert scene.get("cells") is None

        cells = store.sar_matrix(scene["scene_id"])
        assert len(cells) == 25

        matrix, customdata = _sar_grid(cells)
        assert len(matrix) == 5
        assert all(len(row) == 5 for row in matrix)

        # Every cell's sigma0 and classification must survive the reshape, at
        # the right coordinates -- the heatmap addresses cells by (row, col),
        # so a transpose here would render a plausible but wrong image.
        for cell in cells:
            assert matrix[cell.row][cell.col] == cell.sigma0_db
            is_water, classification, row, col = customdata[cell.row][cell.col]
            assert is_water is cell.is_open_water
            assert classification == cell.classification
            assert (row, col) == (cell.row, cell.col)

        # Hover metadata carries the real per-cell classification, not a
        # re-derived threshold duplicated from the classifier.
        assert sorted({m[1] for row in customdata for m in row}) == [
            "consolidated_ice", "open_water",
        ]
    finally:
        store.close()


def test_sar_grid_reports_an_empty_input_rather_than_drawing_a_blank(
    seeded_store: Path,
) -> None:
    """No cells gives an empty result, which the renderer reports."""
    from emperor_space_tracker.dashboard.app import _sar_grid

    assert _sar_grid([]) == ([], [])


def test_sar_grid_leaves_a_gap_for_a_missing_cell(seeded_store: Path) -> None:
    """A hole in the grid plots as a gap, not as open water.

    Filling a missing cell with a low sigma0 would render a lead that was never
    observed -- the same class of invented reading as the one this grid
    previously produced wholesale.
    """
    import math

    from emperor_space_tracker.dashboard.app import _sar_grid
    from emperor_space_tracker.store import Store

    store = Store(str(seeded_store))
    try:
        cells = store.sar_matrix("S1")
        # Drop the centre cell only.
        partial = [
            c for c in cells if not (c.row == 2 and c.col == 2)
        ]
        matrix, customdata = _sar_grid(partial)
        assert matrix[2][2] != matrix[2][2]  # NaN
        assert math.isnan(matrix[2][2])
        assert customdata[2][2][1] == "missing"
        # Everything else is still a real value.
        assert not math.isnan(matrix[0][0])
    finally:
        store.close()


def test_the_heatmap_renders_against_a_populated_store(
    seeded_store: Path,
) -> None:
    """The dashboard draws the heatmap when a scene and its cells both exist."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    # No "no SAR grid available" style warning anywhere in the rendered output.
    warnings = [w.value for w in at.warning]
    assert not any("no SAR grid" in str(w) for w in warnings), warnings


def test_every_plotly_chart_carries_an_explicit_unique_key() -> None:
    """No `plotly_chart` may rely on an auto-generated element id.

    Streamlit derives an element id from the element type and its parameters, so
    two charts built from the same figure collide and raise
    `StreamlitDuplicateElementId` at render time. That is not hypothetical: the
    SAR heatmap used to return early for every scene, so a colony detail page
    only ever drew one chart. Repairing the heatmap put a second one on the page
    and exposed the collision underneath.
    """
    import ast

    tree = ast.parse(APP.read_text())
    offenders: list[int] = []
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "plotly_chart"):
            continue
        total += 1
        if not any(kw.arg == "key" for kw in node.keywords):
            offenders.append(node.lineno)
    assert total >= 4, f"expected several charts, found {total}"
    assert not offenders, f"plotly_chart without a key at lines {offenders}"


def test_the_two_sar_heatmap_call_sites_use_different_key_prefixes() -> None:
    """The fast-ice tab and the colonies tab can draw the same scene.

    Both tabs render in a single pass, so keying only on the scene id would
    collide whenever a page shows the same acquisition twice.
    """
    import ast

    tree = ast.parse(APP.read_text())
    prefixes = {
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_render_sar_imagery"
        for kw in node.keywords
        if kw.arg == "key_prefix" and isinstance(kw.value, ast.Constant)
    }
    assert prefixes == {"fast-ice", "colony"}, prefixes


# --------------------------------------------------------------------------- #
# The authentication gate
# --------------------------------------------------------------------------- #


def test_auth_disabled_renders_normally(isolated_store: Path) -> None:
    """The unauthenticated path must be unchanged, or every existing test moves.

    Loopback with auth off is the coherent default, and it is what the AppTest
    harness exercises, so this pins that the gate is a no-op in that state.

    Uses the isolated fixture deliberately: without it this reads whatever
    ``~/.config`` holds, so an operator's real bind address decides whether the
    dashboard test suite passes.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.tabs) == 5


def test_the_gate_runs_before_the_store_is_opened() -> None:
    """A refused caller must not be able to read a single row.

    Asserted structurally, because the ordering is the whole property: the gate
    has to appear before the `Store(` call, not merely somewhere in `main`.
    """
    import ast

    tree = ast.parse(APP.read_text())
    main = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    called: list[str] = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            called.append(getattr(node.func, "id", "") or getattr(node.func, "attr", ""))
    assert "_require_auth" in called
    assert "Store" in called
    # Source order, not set membership: auth first.
    source = APP.read_text()
    assert source.index("_require_auth(config)") < source.index(
        "Store(config.paths.database_path)"
    )


def test_a_wide_bind_cannot_be_served_unauthenticated() -> None:
    """The gate cannot be bypassed by configuration.

    `est config validate` refuses `host = 0.0.0.0` with `auth.enabled = false`,
    so there is no configuration in which the dashboard is reachable and the
    gate is switched off. This proves the validator actually runs on the path
    that loads the dashboard's config.
    """
    import tempfile

    from emperor_space_tracker.config import (
        load_config,
    )
    from emperor_space_tracker.errors import ConfigError

    wildcard = "0.0.0.0"  # noqa: S104 - the point of the test
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.toml"
        path.write_text(
            f'[paths]\nstate_dir = "{tmp}"\n\n[dashboard]\nhost = "{wildcard}"\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="reachable off"):
            load_config(path, use_user_config=False)


def test_verify_identity_is_what_the_gate_calls() -> None:
    """The gate must not implement its own header handling.

    The shared-secret check has one implementation, in `auth.verify_identity`,
    and it is the one covered by the forgery and fail-closed tests. A second
    implementation in the view layer would be one the tests do not reach.
    """
    import ast

    tree = ast.parse(APP.read_text())
    gate = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_require_auth"
    )
    imported = {
        alias.name
        for node in ast.walk(gate)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "verify_identity" in imported
    # And the gate holds no *secret* comparison of its own. Only string
    # comparisons are forbidden: `identity is None` is a Compare too, and is
    # perfectly legitimate.
    string_comparisons = [
        node
        for node in ast.walk(gate)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(op, ast.Eq | ast.NotEq)
            and isinstance(comparator, ast.Constant)
            and isinstance(comparator.value, str)
            for op, comparator in zip(node.ops, node.comparators, strict=False)
        )
    ]
    assert not string_comparisons, "the gate must delegate the secret comparison"


def test_require_write_allows_write_role_only() -> None:
    """The role ladder is enforced in one place, ready for the first write path.

    The dashboard only reads today, so nothing calls this yet. It exists now so
    that the first write feature uses the shared check rather than inventing a
    per-feature one, which is how a feature ships without a check at all.
    """
    from emperor_space_tracker.auth import Identity, Role
    from emperor_space_tracker.dashboard.app import require_write

    pytest.importorskip("streamlit")
    writer = Identity("kyle", Role.WRITE, "proxy")
    reader = Identity("guest", Role.READ, "proxy")
    assert require_write(writer) is True

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()  # prime st so the error path has somewhere to render
    assert require_write(reader) is False
    # Unauthenticated means loopback-only, so there is nobody to refuse.
    assert require_write(None) is True


def test_the_gate_accepts_a_correctly_signed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accept path, which the AppTest harness cannot reach.

    `st.context.headers` is empty in bare mode, so an end-to-end run can only ever
    exercise the refusal. Poking the context directly covers the other half --
    and the refusal half is worthless without it, because a gate that always said
    "Access denied" would pass every test above.
    """
    pytest.importorskip("streamlit")
    from emperor_space_tracker.auth import AuthStore, Role
    from emperor_space_tracker.config import (
        DashboardAuthConfig,
        load_config,
    )

    pytest.importorskip("plotly")

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[paths]\nstate_dir = "{tmp_path}"\n\n'
        "[dashboard]\nhost = \"127.0.0.1\"\n\n[dashboard.auth]\nenabled = true\n",
        encoding="utf-8",
    )
    config = load_config(config_path, use_user_config=False)
    assert isinstance(config.dashboard.auth, DashboardAuthConfig)

    with AuthStore(tmp_path / "auth.sqlite3") as accounts:
        accounts.add_user("kyle", "a-password", role=Role.WRITE)
        accounts.add_user("guest", "a-password", role=Role.READ)

        # From here on the gate opens the store at this same path.
        import emperor_space_tracker.dashboard.app as app_module
        from emperor_space_tracker.cli import _auth_store_path

        assert _auth_store_path(config) == tmp_path / "auth.sqlite3"

        monkeypatch.setenv("EST_DASHBOARD_PROXY_SECRET", "the-shared-secret")

        headers: dict[str, str] = {
            "X-Emperor-User": "kyle",
            "X-Emperor-Auth": "the-shared-secret",
        }
        monkeypatch.setattr(app_module, "_request_headers", lambda: dict(headers))

        identity = app_module._require_auth(config)
        assert identity is not None
        assert identity.username == "kyle"
        assert identity.can_write
        assert identity.source == "proxy"

        # A read-only account comes back read-only.
        headers.update({"X-Emperor-User": "guest"})
        identity = app_module._require_auth(config)
        assert identity is not None
        assert not identity.can_write

        # A forged identity, with the right name but no secret, is refused.
        # `st.stop()` halts the run, which surfaces as StreamlitAPIException
        # rather than a return value.
        headers.pop("X-Emperor-Auth")
        with pytest.raises(BaseException):  # noqa: B017
            app_module._require_auth(config)


def test_the_gate_refuses_when_the_secret_is_absent_from_the_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dashboard that cannot verify anything must refuse everyone.

    This is the state the node is in between "auth enabled" and "the proxy
    exists", and it has to be a locked door rather than an open one.
    """
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from emperor_space_tracker.auth import AuthStore, Role
    from emperor_space_tracker.config import load_config
    from emperor_space_tracker.dashboard.app import _require_auth

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[paths]\nstate_dir = "{tmp_path}"\n\n'
        "[dashboard]\nhost = \"127.0.0.1\"\n\n[dashboard.auth]\nenabled = true\n",
        encoding="utf-8",
    )
    config = load_config(config_path, use_user_config=False)
    with AuthStore(tmp_path / "auth.sqlite3") as accounts:
        accounts.add_user("kyle", "a-password", role=Role.WRITE)

    monkeypatch.delenv("EST_DASHBOARD_PROXY_SECRET", raising=False)
    import emperor_space_tracker.dashboard.app as app_module

    monkeypatch.setattr(
        app_module,
        "_request_headers",
        lambda: {"X-Emperor-User": "kyle", "X-Emperor-Auth": "anything"},
    )
    with pytest.raises(BaseException):  # noqa: B017
        _require_auth(config)


def test_the_gate_is_a_no_op_when_auth_is_disabled(tmp_path: Path) -> None:
    """Loopback with auth off must not ask for headers it will never get."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from emperor_space_tracker.config import load_config
    from emperor_space_tracker.dashboard.app import _require_auth

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[paths]\nstate_dir = "{tmp_path}"\n\n[dashboard]\nhost = "127.0.0.1"\n',
        encoding="utf-8",
    )
    config = load_config(config_path, use_user_config=False)
    assert _require_auth(config) is None
