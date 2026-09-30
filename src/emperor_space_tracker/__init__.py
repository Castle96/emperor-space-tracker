"""Edge GIS monitoring node for Emperor penguin colonies and fast-ice stability.

The package is split so that the *daemon* half never imports the *frontend* half
and the daemon itself depends on nothing outside the Python standard library:

``config``, ``net``, ``store``, ``sources``, ``alerts``, ``engine``
    Runtime core. Standard library only. Importing any of these is safe on a
    minimal field node with no wheels, no compiler and no site-packages.

``dashboard``
    Streamlit + Plotly frontend. Loaded only by ``emperor-space-tracker
    dashboard``, never by the systemd unit.

The public entry point is :func:`emperor_space_tracker.cli.main`.
"""

from __future__ import annotations

__all__ = [
    "DASHBOARD_AVAILABLE",
    "__version__",
]

__version__ = "0.1.0"


def _probe_frontend() -> bool:
    """Report whether the optional dashboard stack is importable.

    Uses ``importlib.util.find_spec`` rather than a real import so that merely
    installing the package never pulls Streamlit into the daemon's address
    space.
    """
    from importlib.util import find_spec

    return all(find_spec(name) is not None for name in ("streamlit", "plotly"))


DASHBOARD_AVAILABLE: bool = _probe_frontend()
