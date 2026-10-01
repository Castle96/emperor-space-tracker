"""Streamlit frontend for the Emperor Space Tracker.

Futuristic, sci-fi monitoring station aesthetic. Dark deep-space backdrop with
neon cyan/magenta accents, HUD-style panels, and glowing data visualizations.

Two new capabilities beyond the stock Streamlit tab layout:

1. Colony geospatial heatmap. Every known Emperor colony rendered as a glowing
   point on a polar stereographic Antarctic map, sized by population and coloured
   by stability. Click any point to drill in.

2. Colony detail page. For a selected colony the dashboard shows:
   - The latest Sentinel-1 SAR backscatter grid rendered as a heatmap "satellite
     image" of the colony's fast-ice apron (hh/vv sigma0 in dB, dark = open water,
     bright = consolidated ice).
   - A population representation: a dot field where each dot stands for a fixed
     number of breeding pairs, so a 24,000-pair colony and a 200-pair colony are
     visually comparable. Individual penguin photographs are not available from
     any public source; this is an honest derived visualization, not a photo.
   - The full census record: region, population, census year, source, fast-ice
     ratio, last census date, and the curator notes from the SCAR/CCAMLR record.
   - The colony's own sigma0 trend line, so a reader can see whether its fast ice
     is stable, stressed or breaking up.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from emperor_space_tracker.config import Config, load_config
from emperor_space_tracker.store import Store

if TYPE_CHECKING:
    import plotly.graph_objects as go
    import streamlit as st

# At runtime, Streamlit + Plotly are optional extra dependencies. Import them
# here so every helper defined at module scope (e.g. _chart_card, _stability_badge)
# can reference `st` and `go` without needing them threaded through as parameters.
try:
    import plotly.graph_objects as go
    import streamlit as st
except ImportError as exc:
    missing: list[str] = []
    try:
        import streamlit
    except ImportError:
        missing.append("streamlit (dashboard extra: uv add --editable .[dashboard])")
    try:
        import plotly
    except ImportError:
        missing.append("plotly")
    raise RuntimeError(
        "This module requires optional dependencies that are not installed.\n"
        + "\n".join(f"  - {m}" for m in missing)
    ) from exc

_LOG = logging.getLogger("emperor.dashboard")

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

_THEME_CSS: Final[str] = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep: #07070f;
    --bg-surface: #0d0d1a;
    --bg-elevated: #121225;
    --border-default: #1e1e3a;
    --border-glow: #00d4ff;
    --text-primary: #e0e0f0;
    --text-secondary: #8888aa;
    --text-dim: #555577;
    --accent-cyan: #00d4ff;
    --accent-magenta: #ff00aa;
    --accent-amber: #ff8800;
    --accent-green: #00ff88;
    --accent-red: #ff0044;
}

* { box-sizing: border-box; }

body {
    background: var(--bg-deep);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif;
    font-weight: 600;
    color: var(--text-primary);
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

.stApp { background: var(--bg-deep) !important; }
.stSidebar { background: var(--bg-surface) !important; border-right: 1px solid var(--border-default) !important; }

.stSidebar .stSelectbox label {
    color: var(--text-secondary) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 10px !important;
    letter-spacing: 1px !important;
}

.stSidebar .stSelectbox select {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-default) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

.stMetric { background: var(--bg-surface) !important; border: 1px solid var(--border-default) !important; border-radius: 4px !important; }
.stMetric .stMetricValue { font-family: 'Share Tech Mono', monospace !important; color: var(--accent-cyan) !important; font-size: 28px !important; }
.stMetric .stMetricLabel { color: var(--text-secondary) !important; font-family: 'Orbitron', sans-serif !important; font-size: 10px !important; letter-spacing: 1px !important; }

.data-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 4px;
    padding: 16px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
}

.data-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
}

.data-panel.cyan { border-color: rgba(0, 212, 255, 0.3); box-shadow: 0 0 20px rgba(0, 212, 255, 0.15); }
.data-panel.magenta { border-color: rgba(255, 0, 170, 0.3); box-shadow: 0 0 20px rgba(255, 0, 170, 0.15); }
.data-panel.amber { border-color: rgba(255, 136, 0, 0.3); box-shadow: 0 0 15px rgba(255, 136, 0, 0.15); }

.panel-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    font-weight: 500;
    color: var(--accent-cyan);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-default);
}

.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent-green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent-green);
    animation: pulse 2s infinite;
    margin-right: 8px;
    vertical-align: middle;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 2px;
    font-family: 'Orbitron', sans-serif;
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.badge-stable { background: rgba(0, 255, 136, 0.15); color: #00ff88; border: 1px solid rgba(0, 255, 136, 0.3); }
.badge-nominal { background: rgba(0, 212, 255, 0.15); color: #00d4ff; border: 1px solid rgba(0, 212, 255, 0.3); }
.badge-stressed { background: rgba(255, 136, 0, 0.15); color: #ff8800; border: 1px solid rgba(255, 136, 0, 0.3); }
.badge-breached { background: rgba(255, 0, 68, 0.15); color: #ff0044; border: 1px solid rgba(255, 0, 68, 0.3); }
.badge-dispersed { background: rgba(255, 0, 170, 0.15); color: #ff00aa; border: 1px solid rgba(255, 0, 170, 0.3); }

.data-value { font-family: 'Share Tech Mono', monospace; font-size: 14px; color: var(--accent-cyan); }
.data-label { font-family: 'Orbitron', sans-serif; font-size: 9px; color: var(--text-dim); letter-spacing: 1px; text-transform: uppercase; }

.colony-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 4px;
    padding: 14px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.3s ease;
}

.colony-card:hover {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
    transform: translateY(-1px);
}

.colony-card.selected {
    border-color: var(--accent-magenta);
    box-shadow: 0 0 20px rgba(255, 0, 170, 0.25);
}

.colony-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 2px;
}

.colony-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: var(--text-secondary);
}

.row-cols-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.row-cols-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }

@media (max-width: 900px) {
    .row-cols-2, .row-cols-3 { grid-template-columns: 1fr; }
}

.detail-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 9px;
    color: var(--text-dim);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 2px;
}

.detail-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.alert-critical {
    animation: glow-pulse 2s infinite;
}

@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 10px rgba(255, 0, 68, 0.3); }
    50% { box-shadow: 0 0 25px rgba(255, 0, 68, 0.6); }
}

.plot-container { background: var(--bg-surface); border: 1px solid var(--border-default); border-radius: 4px; padding: 4px; margin: 8px 0; }
</style>
"""

# ---------------------------------------------------------------------------
# Shared chart chrome
# ---------------------------------------------------------------------------


def _base_layout(figure: Any, *, height: int = 280) -> Any:
    """Apply the shared chart chrome: dark theme, neon gridlines, Orbitron axes."""
    figure.update_layout(
        height=height,
        margin=dict(l=56, r=20, t=36, b=40),
        paper_bgcolor="rgba(13,13,26,1)",
        plot_bgcolor="rgba(7,7,15,1)",
        font=dict(size=11, color="#8888aa", family="Share Tech Mono, monospace"),
        xaxis=dict(
            gridcolor="rgba(0,212,255,0.08)",
            zerolinecolor="rgba(0,212,255,0.15)",
            tickfont=dict(family="Share Tech Mono, monospace", size=10, color="#555577"),
        ),
        yaxis=dict(
            gridcolor="rgba(0,212,255,0.08)",
            zerolinecolor="rgba(0,212,255,0.15)",
            tickfont=dict(family="Share Tech Mono, monospace", size=10, color="#555577"),
        ),
        showlegend=True,
        legend=dict(
            font=dict(family="Orbitron, sans-serif", size=9, color="#8888aa"),
            bgcolor="rgba(13,13,26,0.8)",
            bordercolor="rgba(30,30,58,0.8)",
            borderwidth=1,
        ),
    )
    return figure


def _chart_card(title: str, figure: Any, *, height: int = 280, key: str | None = None) -> None:
    """Render a chart inside a styled panel.

    ``key`` must be unique per chart in a run. Streamlit derives an element id
    from the element type and its parameters, so two charts built from the same
    figure -- which is exactly what happens once a colony detail page draws both
    its SAR heatmap and its trend -- collide and raise
    ``StreamlitDuplicateElementId``. The SAR heatmap used to hide this by
    returning early with a warning for every scene, so there was only ever one
    chart on the page.
    """
    st.markdown(
        f'<div class="plot-container"><div class="panel-header">{title}</div></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(figure, width="stretch", height=height, key=key)


# ---------------------------------------------------------------------------
# Colony data helpers
# ---------------------------------------------------------------------------


def _colony_label(c: Any) -> str:
    """Return a display label for a colony."""
    population = f"{c['population_estimate']:,}" if c.get("population_estimate") else "n/a"
    return f"{c['name']} ({population})"


def _stability_badge(stability: str) -> str:
    """Return an HTML badge for a stability grade."""
    return f'<span class="badge badge-{stability}">{stability}</span>'


def _colony_hover(colony: Any, stability: str) -> str:
    """Build the map hover card for one colony.

    Fast-ice fields are shown only for a fast-ice breeder. A land-nesting
    colony has no ``fast_ice_ratio`` and no SAR scene, so printing a percentage
    for one would be a fabricated number, and a ``None * 100`` would take the
    whole map down.
    """
    species = colony.get("species") or "unrecorded"
    common = colony.get("common_name") or ""
    taxon = f"{common} &mdash; {species}" if common else species
    monitors = colony.get("monitors_fast_ice", colony.get("breeding_habitat") == "fast_ice")
    if monitors:
        ratio = colony.get("fast_ice_ratio")
        ice_line = (
            f"Fast-ice ratio: {ratio * 100:.0f}%<br>"
            if isinstance(ratio, (int, float))
            else "Fast-ice ratio: unknown<br>"
        )
    else:
        ice_line = (
            f"Breeding habitat: {colony.get('breeding_habitat', 'land')} "
            "(fast-ice monitoring not applicable)<br>"
        )
    return (
        f"<b>{colony['name']}</b><br>"
        f"{taxon}<br>"
        f"Population: {colony.get('population_estimate', 'unknown'):,} pairs<br>"
        f"Stability: {stability}<br>"
        f"{ice_line}"
        f"Coordinates: {colony['latitude']:.3f}, {colony['longitude']:.3f}"
    )


def _sar_grid(
    cells: list[Any],
) -> tuple[list[list[float]], list[list[list[Any]]]]:
    """Reshape stored SAR cells into a square grid plus hover metadata.

    The cells live in the ``sar_cells`` table, not in the ``sar_scenes`` row, so
    they have to be fetched separately with :meth:`Store.sar_matrix` and passed
    in. Reading ``scene["cells"]`` -- which the ``sar_scenes`` table does not
    have -- returned an empty matrix, and the heatmap below therefore rendered
    its "no SAR grid available" warning for every scene ever recorded, while
    the grids themselves sat unread in the database.

    Parameters
    ----------
    cells
        Rows from :meth:`Store.sar_matrix`, each carrying ``row``, ``col``,
        ``sigma0_db``, ``is_open_water`` and ``classification``.

    Returns
    -------
    tuple
        ``(matrix, customdata)``. ``matrix`` is ``n`` rows of ``n`` sigma0
        values in dB; ``customdata`` is the same shape, each entry
        ``[is_open_water, classification, row, col]``. Empty input gives empty
        output, which the caller reports rather than drawing a blank plot.
    """
    if not cells:
        return [], []
    by_position: dict[tuple[int, int], Any] = {(c.row, c.col): c for c in cells}
    rows = max(c.row for c in cells) + 1
    cols = max(c.col for c in cells) + 1
    if (rows, cols) != (cols, rows):
        # A non-square grid would silently shear the image, so say so instead.
        st.warning(f"SAR grid is {rows}x{cols}, not square; heatmap skipped.")
        return [], []

    matrix: list[list[float]] = []
    customdata: list[list[list[Any]]] = []
    for row in range(rows):
        matrix_row: list[float] = []
        meta_row: list[list[Any]] = []
        for col in range(cols):
            cell = by_position.get((row, col))
            if cell is None:
                # A missing cell is not a zero-sigma0 cell. Plot it as NaN so
                # the heatmap leaves a gap rather than inventing ice.
                matrix_row.append(float("nan"))
                meta_row.append([None, "missing", row, col])
                continue
            matrix_row.append(cell.sigma0_db)
            meta_row.append([cell.is_open_water, cell.classification, row, col])
        matrix.append(matrix_row)
        customdata.append(meta_row)
    return matrix, customdata


# ---------------------------------------------------------------------------
# Colony geospatial heatmap
# ---------------------------------------------------------------------------


def _render_colony_map(
    st: Any,
    go: Any,
    colonies: list[Any],
    *,
    center_lat: float = -70.0,
    center_lon: float = 120.0,
) -> None:
    """Render all colonies as glowing points on a dark Antarctic map.

    Points are sized by population and colored by stability. Clicking a point is
    not wired through Plotly click events into Streamlit state directly, so the
    map is a visual complement to the clickable colony cards below it.
    """
    if not colonies:
        st.info("no colonies recorded yet. run `est poll` or `est seed`.")
        return

    # Normalise the incoming list to flat dicts with top-level lat/lon.
    # Colonies may arrive either as Colonies (dataclass with .latitude) or as
    # GeoJSON Feature dicts (lat/lon in geometry.coordinates). Pull whichever
    # shape we have into a uniform flat dict so the rest of this function can
    # index by ["latitude"] / ["longitude"] without caring about provenance.
    flat: list[dict[str, Any]] = []
    for c in colonies:
        if isinstance(c, dict):
            geo = c.get("geometry") or {}
            coords = geo.get("coordinates") if isinstance(geo, dict) else None
            lat = c.get("latitude")
            lon = c.get("longitude")
            if lat is None and lon is None and coords and len(coords) == 2:
                lon, lat = coords[0], coords[1]
            flat.append(
                {
                    **c.get("properties", {}),
                    "colony_id": c.get("properties", {}).get("colony_id") or c.get("colony_id"),
                    "name": c.get("properties", {}).get("name") or c.get("name"),
                    "region": c.get("properties", {}).get("region") or c.get("region"),
                    "population_estimate": c.get("properties", {}).get("population_estimate")
                    or c.get("population_estimate"),
                    "population_year": c.get("properties", {}).get("population_year")
                    or c.get("population_year"),
                    "population_source": c.get("properties", {}).get("population_source")
                    or c.get("population_source"),
                    "fast_ice_ratio": c.get("properties", {}).get("fast_ice_ratio")
                    or c.get("fast_ice_ratio"),
                    "notes": c.get("properties", {}).get("notes") or c.get("notes"),
                    "latitude": lat,
                    "longitude": lon,
                    "stability": c.get("stability", "unknown"),
                }
            )
        else:
            lat = getattr(c, "latitude", None)
            lon = getattr(c, "longitude", None)
            flat.append(
                {
                    "colony_id": getattr(c, "colony_id", ""),
                    "name": getattr(c, "name", ""),
                    "region": getattr(c, "region", "unknown"),
                    "population_estimate": getattr(c, "population_estimate", None),
                    "population_year": getattr(c, "population_year", None),
                    "population_source": getattr(c, "population_source", "unknown"),
                    "fast_ice_ratio": getattr(c, "fast_ice_ratio", None),
                    "notes": getattr(c, "notes", ""),
                    "latitude": lat,
                    "longitude": lon,
                    "stability": "unknown",
                }
            )

    colonies = flat
    lats = [c["latitude"] for c in colonies]
    lons = [c["longitude"] for c in colonies]
    sizes = [max(8, (c.get("population_estimate") or 100) / 500.0) for c in colonies]
    colors: list[str] = []
    hover_texts: list[str] = []
    for c in colonies:
        stability = c.get("stability", "unknown")
        color_map = {
            "stable": "#00ff88",
            "nominal": "#00d4ff",
            "stressed": "#ff8800",
            "breached": "#ff0044",
            "dispersed": "#ff00aa",
        }
        colors.append(color_map.get(stability, "#8888aa"))
        hover_texts.append(_colony_hover(c, stability))

    figure = go.Figure()

    figure.add_trace(
        go.Scattergeo(
            lon=[0],
            lat=[-90],
            mode="markers",
            marker=dict(size=0.1, color="#0d0d1a"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    figure.add_trace(
        go.Scattergeo(
            lon=lons,
            lat=lats,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1.5, color="#00d4ff"),
                opacity=0.9,
                symbol="circle",
            ),
            text=hover_texts,
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            name="colonies",
        )
    )

    figure.update_layout(
        title=dict(
            text="COLONY DISTRIBUTION · ANTARCTICA",
            font=dict(family="Orbitron, sans-serif", size=12, color="#00d4ff"),
            x=0.5,
            y=0.96,
        ),
        geo=dict(
            # Polar stereographic view centred on the South Pole. "polar" is
            # not a Plotly projection type; "stereographic" rotated to lat -90
            # is the equivalent that Plotly validates.
            projection=dict(type="stereographic", rotation=dict(lon=center_lon, lat=-90)),
            showland=True,
            landcolor="#0d0d1a",
            lakecolor="#0d0d1a",
            oceancolor="#07070f",
            subunitcolor="#1e1e3a",
            bgcolor="#07070f",
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=480,
        paper_bgcolor="rgba(7,7,15,1)",
        plot_bgcolor="rgba(7,7,15,1)",
    )

    st.plotly_chart(figure, use_container_width=True, height=480, key="colony-map")


# ---------------------------------------------------------------------------
# SAR backscatter "satellite image"
# ---------------------------------------------------------------------------


def _render_sar_imagery(
    st: Any, go: Any, scene: Any, cells: list[Any], *, key_prefix: str
) -> None:
    """Render a SAR backscatter grid as a heatmap satellite view.

    Dark cells = open water / low sigma0. Bright cells = consolidated ice / high
    sigma0. The color scale runs from deep blue (open water) through purple to
    neon cyan (bright ice), matching the dashboard palette.

    Parameters
    ----------
    cells
        The scene's stored cells from :meth:`Store.sar_matrix`. Passed in
        because they are not part of the ``sar_scenes`` row.
    key_prefix
        Distinguishes this element from any other on the page. Streamlit runs
        every tab in one pass, so the fast-ice tab and the colonies tab can draw
        the *same* scene; keying on the scene alone then collides.
    """
    matrix, customdata = _sar_grid(cells)
    if not matrix:
        st.warning(
            "no SAR grid stored for this scene. The grid lives in the sar_cells "
            "table; if this persists after a poll, the scene metadata was "
            "written without its cells."
        )
        return

    size = len(matrix)
    # The posting is a per-node config fact (`sar.resolution_meters`), not a
    # per-scene one -- the sar_scenes row has no such column, so a previous
    # `scene.get("resolution_meters", 100)` always printed a hardcoded 100 that
    # happened to be right by coincidence. Report the grid, which is measured.
    extent = f"{size}x{size} cells"

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            colorscale=[
                [0.0, "#0a1628"],
                [0.2, "#0d2137"],
                [0.4, "#1a1a4e"],
                [0.6, "#3d1a6e"],
                [0.8, "#00d4ff"],
                [1.0, "#ffffff"],
            ],
            zmin=-22.0,
            zmax=-2.0,
            customdata=customdata,
            hovertemplate=(
                "Sigma0: %{z:.1f} dB<br>"
                "Surface: %{customdata[1]}<br>"
                "Cell: %{customdata[2]}, %{customdata[3]}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=dict(
            text=f"SAR BACKSCATTER · {scene.get('scene_id', 'scene')}",
            font=dict(family="Orbitron, sans-serif", size=11, color="#00d4ff"),
            x=0.5,
            y=0.97,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,212,255,0.05)",
            tickmode="array",
            tickvals=list(range(0, size, max(1, size // 5))),
            tickfont=dict(family="Share Tech Mono, monospace", size=9, color="#555577"),
            title=dict(
                text="COLUMN", font=dict(family="Orbitron, sans-serif", size=8, color="#555577")
            ),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,212,255,0.05)",
            tickmode="array",
            tickvals=list(range(0, size, max(1, size // 5))),
            tickfont=dict(family="Share Tech Mono, monospace", size=9, color="#555577"),
            title=dict(
                text="ROW", font=dict(family="Orbitron, sans-serif", size=8, color="#555577")
            ),
            autorange="reversed",
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text="σ⁰ dB",
                font=dict(family="Share Tech Mono, monospace", size=9, color="#8888aa"),
            ),
            tickfont=dict(family="Share Tech Mono, monospace", size=9, color="#555577"),
            outlinewidth=0,
            thickness=12,
            len=160,
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=420,
        paper_bgcolor="rgba(7,7,15,1)",
        plot_bgcolor="rgba(7,7,15,1)",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        height=420,
        key=f"sar-grid-{key_prefix}-{scene['scene_id']}",
    )

    # Caption with physical interpretation
    mean_db = scene.get("mean_db")
    water_pct = scene.get("open_water_fraction", 0) * 100
    stability = scene.get("stability", "unknown")
    provenance = scene.get("provenance", "")
    synthetic = provenance.startswith("synthetic")

    st.markdown(
        f"""
        <div class="data-panel cyan" style="margin-top:4px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div class="data-label">MEAN SIGMA0</div>
                    <div class="data-value">{mean_db:.2f} dB</div>
                </div>
                <div>
                    <div class="data-label">OPEN WATER</div>
                    <div class="data-value">{water_pct:.1f}%</div>
                </div>
                <div>
                    <div class="data-label">STABILITY</div>
                    {_stability_badge(stability)}
                </div>
                <div>
                    <div class="data-label">POSTING</div>
                    <div class="data-value">{extent}</div>
                </div>
            </div>
            {'<div style="margin-top:8px;font-size:11px;color:#b388ff;font-style:italic;">⚠ Simulated scene — exercises the pipeline, not a real observation.</div>' if synthetic else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Penguin population representation
# ---------------------------------------------------------------------------


def _colony_seed(colony_id: str) -> int:
    """Return a deterministic PRNG seed for a colony's dot layout.

    NOTE: the builtin hash() is salted per process (PYTHONHASHSEED), so it
    would reshuffle the colony on every rerun. MD5 is deterministic here;
    it seeds a PRNG layout, not a security decision.
    """
    digest = hashlib.md5(colony_id.encode(), usedforsecurity=False).hexdigest()
    return int(digest, 16) % (2**31)


def _render_penguin_population(colony: dict[str, Any]) -> None:
    """Render a dot field representing the colony's breeding-pair population.

    Each dot stands for a configurable number of pairs. The dots are arranged in
    a clustered pattern resembling a real colony distribution on fast ice. This is
    an honest derived visualization — individual penguin photographs are not
    available from any public source, and this project does not collect them.
    """
    population = colony.get("population_estimate")
    if population is None or population <= 0:
        st.warning("no population estimate available for this colony.")
        return

    dots_per_unit = 1  # one dot per breeding pair — scale down for large colonies
    if population > 5000:
        dots_per_unit = population / 2000.0  # ~2000 dots for a large colony
    elif population > 1000:
        dots_per_unit = population / 800.0

    n_dots = max(50, min(3000, int(population / dots_per_unit)))
    actual_per_dot = population / n_dots

    # Cluster the dots into sub-groups to resemble a real colony spread
    import random

    random.seed(_colony_seed(str(colony.get("colony_id", "unknown"))))

    clusters = max(3, min(12, n_dots // 200))
    cluster_centers: list[tuple[float, float]] = []
    for _ in range(clusters):
        cluster_centers.append((random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)))

    dots: list[tuple[float, float, str]] = []
    for _ in range(n_dots):
        ci = random.randrange(len(cluster_centers))
        cx, cy = cluster_centers[ci]
        spread = random.uniform(0.02, 0.08)
        x = max(0.02, min(0.98, cx + random.gauss(0, spread)))
        y = max(0.02, min(0.98, cy + random.gauss(0, spread)))
        dots.append((x, y, colony.get("colony_id", "???")))

    # Render as a scatter plot with a dark background
    figure = go.Figure()

    # Background grid
    figure.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Penguin dots
    figure.add_trace(
        go.Scatter(
            x=[d[0] for d in dots],
            y=[d[1] for d in dots],
            mode="markers",
            marker=dict(
                size=4,
                color="#00d4ff",
                line=dict(width=0.5, color="#00ff88"),
                opacity=0.85,
                symbol="circle",
            ),
            text=[
                f"{colony['name']}<br>{du} breeding pairs per dot<br>{total:,} total"
                for total in [population]
                for du in [actual_per_dot]
            ],
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            name="breeding pairs",
        )
    )

    figure.update_layout(
        title=dict(
            text=f"POPULATION REPRESENTATION · {colony['name']}",
            font=dict(family="Orbitron, sans-serif", size=11, color="#00d4ff"),
            x=0.5,
            y=0.97,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,212,255,0.05)",
            range=[0, 1],
            showticklabels=False,
            title=dict(text="", font=dict(size=1)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,212,255,0.05)",
            range=[0, 1],
            showticklabels=False,
            title=dict(text="", font=dict(size=1)),
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=340,
        paper_bgcolor="rgba(7,7,15,1)",
        plot_bgcolor="rgba(7,7,15,1)",
    )

    st.plotly_chart(figure, use_container_width=True, height=340, key="population-field")

    st.markdown(
        f"""
        <div class="data-panel amber" style="margin-top:4px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div class="data-label">BREEDING PAIRS</div>
                    <div class="data-value">{population:,}</div>
                </div>
                <div>
                    <div class="data-label">DOTS SHOWN</div>
                    <div class="data-value">{n_dots:,}</div>
                </div>
                <div>
                    <div class="data-label">PER DOT</div>
                    <div class="data-value">{actual_per_dot:,.0f} pairs</div>
                </div>
                <div>
                    <div class="data-label">CENSUS YEAR</div>
                    <div class="data-value">{colony.get("population_year", "unknown")}</div>
                </div>
            </div>
            <div style="margin-top:8px;font-size:11px;color:#555577;">
                Each dot represents {actual_per_dot:,.0f} breeding pairs. Total population: {population:,} pairs
                ({population * 2:,} individuals). This is a derived visualization — individual penguin
                photographs are not available from any public source.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Colony detail page
# ---------------------------------------------------------------------------


def _render_colony_detail(st: Any, go: Any, store: Store, colony: dict[str, Any]) -> None:
    """Render the full detail page for one colony."""
    cid = colony.get("colony_id", "???")
    st.markdown(
        f'<div style="font-family:Orbitron,sans-serif;font-size:18px;font-weight:700;color:#e0e0f0;letter-spacing:1px;margin-bottom:4px;">{colony["name"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#8888aa;margin-bottom:12px;">{cid} · {colony.get("region", "unknown")}</div>',
        unsafe_allow_html=True,
    )

    # Top row: key stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="data-label">POPULATION</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="data-value" style="font-size:18px;">{colony.get("population_estimate", "n/a"):,}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown('<div class="data-label">FAST-ICE RATIO</div>', unsafe_allow_html=True)
        fr = colony.get("fast_ice_ratio")
        fr_text = f"{fr * 100:.0f}%" if fr is not None else "n/a"
        fr_color = "#00ff88" if (fr or 0) >= 0.8 else ("#ff8800" if (fr or 0) >= 0.6 else "#ff0044")
        st.markdown(
            f'<div class="data-value" style="font-size:18px;color:{fr_color};">{fr_text}</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown('<div class="data-label">LAST CENSUS</div>', unsafe_allow_html=True)
        census = colony.get("last_census_at")
        st.markdown(
            f'<div class="data-value">{census.isoformat() if census else colony.get("population_year", "n/a")}</div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown('<div class="data-label">STABILITY</div>', unsafe_allow_html=True)
        stability = colony.get("stability", "unknown")
        st.markdown(_stability_badge(stability), unsafe_allow_html=True)

    st.divider()

    # SAR imagery + population representation side by side
    latest_scene = store.recent_sar_scenes(limit=1, colony_id=cid)
    if latest_scene:
        scene = latest_scene[0]
        col_a, col_b = st.columns(2)
        with col_a:
            _render_sar_imagery(
                st, go, scene, store.sar_matrix(scene["scene_id"]), key_prefix="colony"
            )
        with col_b:
            _render_penguin_population(colony)
    else:
        st.info(
            f"no SAR imagery yet for {colony['name']}. run `est poll` to fetch Sentinel-1 backscatter."
        )

    st.divider()

    # Facts panel
    st.markdown('<div class="panel-header">COLONY FACTS</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="data-panel magenta">
            <div class="row-cols-2">
                <div>
                    <div class="detail-label">Region</div>
                    <div class="detail-value">{colony.get("region", "unknown")}</div>
                </div>
                <div>
                    <div class="detail-label">Latitude</div>
                    <div class="detail-value">{colony.get("latitude", 0):.4f}°S</div>
                </div>
                <div>
                    <div class="detail-label">Longitude</div>
                    <div class="detail-value">{colony.get("longitude", 0):.4f}°E</div>
                </div>
                <div>
                    <div class="detail-label">Census Source</div>
                    <div class="detail-value" style="font-size:11px;">{colony.get("population_source", "unknown")}</div>
                </div>
                <div>
                    <div class="detail-label">Population Estimate</div>
                    <div class="detail-value">{colony.get("population_estimate", "unknown"):,} breeding pairs</div>
                </div>
                <div>
                    <div class="detail-label">Census Year</div>
                    <div class="detail-value">{colony.get("population_year", "unknown")}</div>
                </div>
                <div>
                    <div class="detail-label">Fast-Ice Ratio</div>
                    <div class="detail-value">{colony.get("fast_ice_ratio", 0) * 100:.0f}% of season</div>
                </div>
                <div>
                    <div class="detail-label">Provenance</div>
                    <div class="detail-value" style="font-size:11px;">{colony.get("population_source", "unknown")}</div>
                </div>
            </div>
            <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border-default);">
                <div class="detail-label">Notes</div>
                <div style="font-family:Inter,sans-serif;font-size:13px;color:#c0c0d0;line-height:1.5;font-style:italic;">
                    {colony.get("notes", "no notes available.")}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Colony sigma0 trend
    trend = store.sigma0_trend(cid, limit=24)
    if trend:
        _render_colony_trend(st, go, trend, colony["name"])


def _render_colony_trend(st: Any, go: Any, trend: list[dict[str, Any]], name: str) -> None:
    """Render a colony's sigma0 trend line."""
    times = [r["observed_at"] for r in trend]
    means = [r["mean_db"] for r in trend]
    waters = [r["open_water_fraction"] * 100 for r in trend]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=times,
            y=means,
            mode="lines+markers",
            name="mean σ⁰",
            line=dict(color="#00d4ff", width=2),
            marker=dict(size=5, color="#00d4ff"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=waters,
            mode="lines",
            name="open water %",
            line=dict(color="#ff8800", width=1.5, dash="dot"),
            yaxis="y2",
        )
    )

    figure.update_layout(
        title=dict(
            text=f"SIGMA0 TREND · {name}",
            font=dict(family="Orbitron, sans-serif", size=11, color="#00d4ff"),
            x=0.5,
            y=0.97,
        ),
        yaxis=dict(title="σ⁰ (dB)", color="#00d4ff"),
        yaxis2=dict(
            title="open water %",
            overlaying="y",
            side="right",
            color="#ff8800",
            showgrid=False,
        ),
        legend=dict(
            font=dict(family="Orbitron, sans-serif", size=9, color="#8888aa"),
            bgcolor="rgba(13,13,26,0.8)",
        ),
        margin=dict(l=50, r=50, t=50, b=40),
        height=240,
        paper_bgcolor="rgba(7,7,15,1)",
        plot_bgcolor="rgba(7,7,15,1)",
    )

    _chart_card(f"sigma0 trend · {name}", figure, height=240, key=f"colony-trend-{name}")


# ---------------------------------------------------------------------------
# Colony cards (clickable list)
# ---------------------------------------------------------------------------


def _render_colony_list(
    st: Any, colonies: list[dict[str, Any]], selected_id: str | None
) -> str | None:
    """Render clickable colony cards. Returns the selected colony_id."""
    selected: str | None = selected_id
    for c in sorted(colonies, key=lambda x: -(x.get("population_estimate") or 0)):
        cid = c.get("colony_id", "???")
        is_selected = cid == selected
        stability = c.get("stability", "unknown")
        pop = c.get("population_estimate")
        pop_text = f"{pop:,}" if pop else "n/a"
        fr = c.get("fast_ice_ratio")
        fr_text = f"{fr * 100:.0f}%" if fr is not None else "n/a"

        card_html = f"""
        <div class="colony-card {"selected" if is_selected else ""}" onclick="window.location.hash='#{cid}'">
            <div style="display:flex;justify-content:space-between;align-items:start;">
                <div>
                    <div class="colony-name">{c["name"]}</div>
                    <div class="colony-sub">{c.get("region", "unknown")} · {c.get("latitude", 0):.3f}S, {c.get("longitude", 0):.3f}E</div>
                </div>
                <div style="text-align:right;min-width:80px;">
                    {_stability_badge(stability)}
                    <div style="margin-top:4px;font-family:Share Tech Mono,monospace;font-size:11px;color:#00d4ff;">{pop_text} pairs</div>
                </div>
            </div>
            <div style="display:flex;gap:16px;margin-top:6px;">
                <span style="font-family:Orbitron,sans-serif;font-size:9px;color:#555577;letter-spacing:1px;">FAST-ICE</span>
                <span style="font-family:Share Tech Mono,monospace;font-size:11px;color:{"#00ff88" if (fr or 0) >= 0.8 else "#ff8800" if (fr or 0) >= 0.6 else "#ff0044"};">{fr_text}</span>
                <span style="font-family:Orbitron,sans-serif;font-size:9px;color:#555577;letter-spacing:1px;margin-left:auto;">CENSUS</span>
                <span style="font-family:Share Tech Mono,monospace;font-size:11px;color:#8888aa;">{c.get("population_year", "n/a")}</span>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    return selected


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _tab_overview(st: Any, store: Store) -> None:
    """Futuristic rewrite of the overview tab."""
    latest = store.latest_space_weather()
    health = store.health_summary(minutes=180)
    alerts = store.recent_alerts(limit=8)
    stats = store.stats()

    # Live indicator
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
        '<span style="font-family:Orbitron,sans-serif;font-size:10px;color:#00d4ff;letter-spacing:2px;">EMONITOR · EMPEROR SPACE TRACKER</span>'
        '<span class="live-dot"></span><span style="font-family:Share Tech Mono,monospace;font-size:10px;color:#00ff88;">SYSTEM ONLINE</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Headline metrics
    st.markdown(
        '<div class="panel-header">SENSOR ARRAY · CURRENT CONDITIONS</div>', unsafe_allow_html=True
    )

    cols = st.columns(4)
    with cols[0]:
        if latest is None:
            st.markdown('<div class="data-label">SOLAR WIND</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="data-value" style="font-size:20px;color:#555577;">NO DATA</div>',
                unsafe_allow_html=True,
            )
        else:
            st.metric("SOLAR WIND SPEED", f"{latest['speed_kms']:.0f}", "km/s", delta_color="off")
    with cols[1]:
        st.metric(
            "PLANETARY K",
            f"{latest['kp_value']:.1f}" if latest else "NO DATA",
            "G-scale" if latest and latest.get("storm_active") else "quiet",
            delta_color="off",
        )
    with cols[2]:
        bad = [h for h in health if not h["ok"]]
        st.metric(
            "SOURCES HEALTHY", f"{len(health) - len(bad)}/{len(health)}", None, delta_color="off"
        )
    with cols[3]:
        st.metric("ALERTS (RECENT)", str(len(alerts)), None, delta_color="off")

    # Source health table
    st.markdown(
        '<div class="panel-header" style="margin-top:16px;">SOURCE HEALTH · TELEMETRY</div>',
        unsafe_allow_html=True,
    )
    if not health:
        st.info("no health records in the last three hours. run `est poll`.")
    else:
        health_data = []
        for h in health:
            status_color = {"ok": "#00ff88", "degraded": "#ff8800", "down": "#ff0044"}[h["status"]]
            health_data.append(
                {
                    "source": h["source"],
                    "status": f'<span style="color:{status_color};font-family:Orbitron,sans-serif;font-size:9px;">{h["status"].upper()}</span>',
                    "latency": f"{h['latency_ms']} ms",
                    "records": h["record_count"],
                    "detail": h["detail"],
                    "seen": _ago(h["observed_at"]),
                }
            )
        st.dataframe(health_data, hide_index=True, use_container_width=True)

    # Stored rows
    st.markdown(
        '<div class="panel-header" style="margin-top:16px;">ON-DISK ARCHIVE</div>',
        unsafe_allow_html=True,
    )
    total_rows = sum(s.rows for s in stats)
    st.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#8888aa;">'
        f'Total rows: <span style="color:#00d4ff;font-size:14px;">{total_rows:,}</span> · '
        f'Database: <span style="color:#00d4ff;">{store.size_bytes() / (1024 * 1024):.1f} MiB</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#8888aa;line-height:1.6;'
        'padding:10px 12px;border:1px solid var(--border-default);border-radius:4px;margin-top:16px;">'
        "Every figure on this dashboard is a query against the same SQLite file the daemon "
        "writes. The thresholds live in the alert rules, the surface classifications in the "
        "SAR backend, and the presence/absence semantics in the biological source — none of "
        "them are restated here, because a view layer that re-derives its own thresholds is "
        "how a dashboard ends up disagreeing with the process that produced the data."
        "</div>",
        unsafe_allow_html=True,
    )


def _tab_colony_map(st: Any, go: Any, store: Store) -> None:
    """Geospatial heatmap of all colonies."""
    colonies = store.colonies(limit=200)
    colony_dicts = [c.as_geojson() for c in colonies]

    # Add stability from recent SAR scenes
    for gd in colony_dicts:
        scenes = store.recent_sar_scenes(limit=1, colony_id=gd["properties"]["colony_id"])
        gd["properties"]["stability"] = scenes[0]["stability"] if scenes else "unknown"
        gd["properties"]["population_estimate"] = gd["properties"]["population_estimate"]
        gd["properties"]["fast_ice_ratio"] = gd["properties"]["fast_ice_ratio"]
        gd["properties"]["name"] = gd["properties"]["name"]

    _render_colony_map(st, go, colony_dicts)


def _tab_colony_detail(st: Any, go: Any, store: Store, colonies: list[Any]) -> None:
    """Interactive colony detail: map + clickable list + detail page."""
    # Sidebar: colony picker
    colony_ids = [c.colony_id for c in colonies]
    labels = {c.colony_id: f"{c.name} ({c.population_estimate or 0:,})" for c in colonies}

    selected = st.selectbox(
        "SELECT COLONY",
        options=colony_ids,
        format_func=lambda cid: labels.get(cid, cid),
        key="colony_picker",
        label_visibility="collapsed",
    )

    if selected:
        colony_obj = next((c for c in colonies if c.colony_id == selected), None)
        if colony_obj:
            scene = store.recent_sar_scenes(limit=1, colony_id=colony_obj.colony_id)
            stability = scene[0]["stability"] if scene else "unknown"
            # Build a dict with all the fields the detail page expects
            detail = {
                "colony_id": colony_obj.colony_id,
                "name": colony_obj.name,
                "region": colony_obj.region,
                "latitude": colony_obj.latitude,
                "longitude": colony_obj.longitude,
                "population_estimate": colony_obj.population_estimate,
                "population_year": colony_obj.population_year,
                "population_source": colony_obj.population_source,
                "fast_ice_ratio": colony_obj.fast_ice_ratio,
                "last_census_at": colony_obj.last_census_at,
                "notes": colony_obj.notes,
                "stability": stability,
            }
            _render_colony_detail(st, go, store, detail)

    # Mini map at bottom
    st.markdown(
        '<div class="panel-header" style="margin-top:24px;">COLONY DISTRIBUTION · MAP</div>',
        unsafe_allow_html=True,
    )
    _tab_colony_map(st, go, store)


def _tab_space_weather(st: Any, go: Any, store: Store, hours: int) -> None:
    """Space weather tab with futuristic styling."""
    rows = store.space_weather_series(hours=hours, limit=4000)
    if not rows:
        st.info(f"no space weather observations in the last {hours} hours.")
        return

    frame = []
    for r in rows:
        frame.append(
            {
                "t": r["observed_at"],
                "speed": r["speed_kms"],
                "density": r["density_per_cm3"],
                "bz": r["bz_gsm_nt"],
                "bt": r["bt_nt"],
                "kp": r["kp_value"],
                "f107": r["f107_sfu"],
                "storm": bool(r["storm_active"]),
            }
        )

    left, right = st.columns(2)

    with left:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=[r["t"] for r in frame],
                y=[r["speed"] for r in frame],
                name="SPEED",
                line=dict(color="#00d4ff", width=1.5),
            )
        )
        _chart_card("SOLAR WIND SPEED · km/s", figure, height=240, key="sw-speed")

    with right:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=[r["t"] for r in frame],
                y=[r["bz"] for r in frame],
                name="BZ GSM",
                line=dict(color="#ff00aa", width=1.5),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[r["t"] for r in frame],
                y=[r["bt"] for r in frame],
                name="BT",
                line=dict(color="#ff8800", width=1, dash="dot"),
            )
        )
        figure.add_hline(y=0, line=dict(color="rgba(0,212,255,0.2)", width=1))
        _chart_card(
            "INTERPLANETARY MAGNETIC FIELD · nT", figure, height=240, key="sw-imf"
        )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[r["t"] for r in frame],
            y=[r["kp"] for r in frame],
            name="Kp",
            line=dict(color="#9d4EDD", width=2),
            fill="tozeroy",
            fillcolor="rgba(157,78,221,0.12)",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[r["t"] for r in frame],
            y=[r["density"] for r in frame],
            name="DENSITY",
            line=dict(color="#00ff88", width=2),
            yaxis="y2",
        )
    )
    figure.update_layout(
        yaxis=dict(title="Kp", color="#9d4EDD"),
        yaxis2=dict(
            title="Density (p/cm³)",
            overlaying="y",
            side="right",
            showgrid=False,
            color="#00ff88",
        ),
    )
    _chart_card("Kp INDEX · SOLAR WIND DENSITY", figure, height=220, key="sw-kp")

    degraded = {d for r in rows for d in (r["degraded"] or [])}
    if degraded:
        st.markdown(
            f'<div class="data-panel amber"><div class="panel-header">DEGRADED FIELDS</div>'
            f'<span style="font-family:Share Tech Mono,monospace;color:#ff8800;">{" · ".join(sorted(degraded))}</span></div>',
            unsafe_allow_html=True,
        )


def _tab_fast_ice(st: Any, go: Any, store: Store) -> None:
    """Fast ice tab showing per-colony backscatter trend."""
    colonies = store.colonies(limit=50)
    if not colonies:
        st.info("no colonies recorded yet. run `est poll` or `est seed`.")
        return

    chosen = st.selectbox(
        "SELECT COLONY",
        options=[c.colony_id for c in colonies],
        format_func=lambda cid: next((c.name for c in colonies if c.colony_id == cid), cid),
        label_visibility="collapsed",
    )

    trend = store.sigma0_trend(chosen, limit=24)
    if not trend:
        st.info(f"no backscatter scenes recorded for {chosen}.")
        return

    latest_scene = store.recent_sar_scenes(limit=1, colony_id=chosen)

    col1, col2 = st.columns([2, 1])
    with col1:
        times = [r["observed_at"] for r in trend]
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=times,
                y=[r["mean_db"] for r in trend],
                name="mean σ⁰",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=5),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=[r["min_db"] for r in trend],
                name="min",
                line=dict(width=1, color="#555577"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=[r["max_db"] for r in trend],
                name="max",
                line=dict(width=1, color="#555577"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=[r["open_water_fraction"] * -20 for r in trend],
                name="open water (scaled)",
                line=dict(color="#00ff88", width=1.5, dash="dot"),
            )
        )
        _chart_card(
            f"BACKSCATTER TREND · {chosen}", figure, height=280, key=f"ice-trend-{chosen}"
        )

    with col2:
        st.markdown('<div class="panel-header">LATEST SCENE</div>', unsafe_allow_html=True)
        if latest_scene:
            scene = latest_scene[0]
            st.markdown(
                f'<div class="data-label">SCENE ID</div><div class="data-value" style="font-size:11px;">{scene["scene_id"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="data-label">MEAN SIGMA0</div><div class="data-value">{scene["mean_db"]:.2f} dB</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="data-label">OPEN WATER</div><div class="data-value">{scene["open_water_fraction"] * 100:.1f}%</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="data-label">STABILITY</div>{_stability_badge(scene["stability"])}',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="data-label">PROVENANCE</div><div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#b388ff;">{scene["provenance"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="color:#555577;font-family:Share Tech Mono,monospace;">no scene data</div>',
                unsafe_allow_html=True,
            )

        with st.expander("SAR BACKSCATTER IMAGE"):
            if latest_scene:
                _render_sar_imagery(
                    st,
                    go,
                    latest_scene[0],
                    store.sar_matrix(latest_scene[0]["scene_id"]),
                    key_prefix="fast-ice",
                )


def _tab_colonies(st: Any, go: Any, store: Store) -> None:
    """Colonies tab: interactive map + detail. Replaces the plain table."""
    colonies = store.colonies(limit=200)
    if not colonies:
        st.info("no colonies recorded yet.")
        return

    _tab_colony_detail(st, go, store, colonies)


def _tab_alerts(st: Any, store: Store) -> None:
    """Alerts tab with futuristic styling."""
    alerts = store.recent_alerts(limit=100)
    if not alerts:
        st.markdown(
            '<div class="data-panel cyan"><div class="panel-header">ALERT LOG</div>'
            '<div style="font-family:Share Tech Mono,monospace;color:#00ff88;font-size:14px;">NO ALERTS FIRED · SYSTEM NOMINAL</div></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="panel-header">ALERT LOG · DISPATCH HISTORY</div>', unsafe_allow_html=True
    )
    alert_data = []
    for a in alerts:
        delivered = "SENT" if a.delivered else "LOGGED"
        color = "#00ff88" if a.delivered else "#ff8800"
        alert_data.append(
            {
                "time": a.fired_at.strftime("%Y-%m-%d %H:%M UTC"),
                "severity": f'<span style="color:{"#ff0044" if a.severity.value in ("severe", "critical") else "#ff8800" if a.severity.value == "warning" else "#00d4ff"};font-family:Orbitron,sans-serif;font-size:9px;">{a.severity.value.upper()}</span>',
                "rule": a.rule_id,
                "title": a.title,
                "status": f'<span style="color:{color};font-family:Orbitron,sans-serif;font-size:9px;">{delivered}</span>',
                "error": a.delivery_error or "",
            }
        )
    st.dataframe(alert_data, hide_index=True, use_container_width=True)

    undelivered = [a for a in alerts if not a.delivered]
    if undelivered:
        st.markdown(
            f'<div class="data-panel alert-critical" style="border-color:rgba(255,0,68,0.3);">'
            f'<div class="panel-header" style="color:#ff0044;">UNDELIVERED ALERTS</div>'
            f'<span style="font-family:Share Tech Mono,monospace;color:#ff0044;">{len(undelivered)} alert(s) not delivered — check webhook URL and logs</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


def _ago(moment: datetime | None) -> str:
    """Render how long ago ``moment`` was."""
    if moment is None:
        return "never"
    delta = datetime.now(UTC) - moment
    seconds = delta.total_seconds()
    if seconds < 0:
        return "in the future"
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds / 60)}m ago"
    if seconds < 172800:
        return f"{int(seconds / 3600)}h ago"
    return f"{int(seconds / 86400)}d ago"


# ---------------------------------------------------------------------------
# Optional-dependency loader
# ---------------------------------------------------------------------------


def _require_extras() -> tuple[Any, Any]:
    """Load Streamlit + Plotly, raising a clear message when missing."""
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except ImportError as exc:
        missing = []
        try:
            import streamlit
        except ImportError:
            missing.append("streamlit (dashboard extra: uv add --editable .[dashboard])")
        try:
            import plotly
        except ImportError:
            missing.append("plotly")
        raise RuntimeError(
            "This module requires optional dependencies that are not installed.\n"
            + "\n".join(f"  - {m}" for m in missing)
        ) from exc
    return st, go


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
#
# The gate runs before the store is opened and before a single row is read, so a
# refused caller never touches the observation database at all.
#
# There is no login form here, and that is deliberate. Streamlit 1.64 removed
# `[server] password` and gives scripts no cookie *setter* -- `st.context.cookies`
# is read-only -- so an app-side session cannot survive a browser refresh through
# any supported API. A session mechanism that cannot set its own cookie is one
# that fails open the first time the websocket is re-established, which is the
# worst possible failure for a security control. Authentication therefore
# belongs to a reverse proxy that holds a connection-level credential, and this
# gate's job is the part a proxy cannot do: decide whether the identity it
# forwarded should be trusted, and what that identity may see.
#
# See `emperor_space_tracker.auth` for why the identity header is trusted only
# when it carries a matching shared secret, and why the absence of a secret is
# a denial rather than a fallback to trusting the header.

_IDENTITY_HEADER = "X-Emperor-User"
_SECRET_HEADER = "X-Emperor-Auth"  # noqa: S105 - a header name, not a secret


def _request_headers() -> dict[str, str]:
    """Return the current request's headers.

    A seam, not an abstraction: `st.context.headers` is empty in Streamlit's
    bare mode, so without this the gate's *accept* path is unreachable from the
    test suite and only ever the refusal can be proven. Wrapping the one call
    means both halves are testable.
    """
    return dict(st.context.headers)


def _esc(value: object) -> str:
    """Escape a value for interpolation into raw HTML or markdown.

    The username is not ours: it arrives in a request header from the proxy,
    and under Cloudflare Access it is an email address. Rendering it into an
    ``unsafe_allow_html`` block unescaped would make any future path that
    reaches this code with an untrusted value an HTML injection, so escape at
    the point of use rather than trusting every caller upstream.
    """
    return html.escape(str(value), quote=True)


def _require_auth(config: Config) -> Any | None:
    """Resolve the caller, refusing anyone not authenticated.

    Returns the caller's :class:`~emperor_space_tracker.auth.Identity`, or
    ``None`` when authentication is switched off. On refusal the Streamlit run
    is halted immediately, so no data is rendered and the script never reaches
    the store.

    Parameters
    ----------
    config
        The loaded configuration.

    Returns
    -------
    Any | None
        The authenticated identity, or ``None`` if auth is disabled.
    """
    import os as _os

    from emperor_space_tracker.auth import AuthStore
    from emperor_space_tracker.auth import verify_identity as _verify

    if not config.dashboard.auth.enabled:
        return None

    auth_cfg = config.dashboard.auth
    expected = _os.environ.get(auth_cfg.proxy_secret_env)

    # The account store is optional. When it has accounts, roles come from here
    # even though the login happened at the proxy -- so the proxy controls *who*
    # may connect and this node controls *what* they may do, which is what makes a
    # permissive proxy safe to point at this app. When it is absent or empty, the
    # proxy is the only authorisation layer and every authenticated caller is a
    # writer.
    #
    # That is the right default for the dashboard as it stands, because the
    # dashboard only reads. The first write path is where it stops being right,
    # and `est auth add-user <name> --role read` is the fix.
    users: dict[str, Any] = {}
    try:
        from emperor_space_tracker.cli import _auth_store_path

        auth_db = _auth_store_path(config)
        if auth_db.is_file():
            with AuthStore(auth_db) as accounts:
                users = {u.username: u.role for u in accounts.users() if not u.disabled_at}
        else:
            _LOG.info(
                "no auth account store at %s; the proxy is the only authorisation layer "
                "and every authenticated caller is granted write access",
                auth_db,
            )
    except Exception as exc:
        st.error(
            "Dashboard authentication is enabled but the account store could not be "
            f"read ({type(exc).__name__}: {exc}). Refusing the connection rather than "
            "serving unauthenticated."
        )
        st.stop()

    headers = _request_headers()
    identity = _verify(
        headers.get(_IDENTITY_HEADER),
        headers.get(_SECRET_HEADER),
        expected_secret=expected,
        users=users,
    )

    if identity is None:
        # Distinguish "no secret configured" from "wrong identity", because the
        # fix is different and the first is an operator error rather than an
        # attack. The detail is shown only to a loopback caller: a remote host
        # learns the least that still lets it self-diagnose.
        reason = (
            f"${auth_cfg.proxy_secret_env} is not set on this node, so no proxied "
            "identity can be verified."
            if not expected
            else "The request did not carry a valid authenticated identity."
        )
        st.error("Access denied.")
        st.caption(reason)
        st.caption(
            "The dashboard is meant to sit behind a reverse proxy that authenticates the "
            "connection. A direct connection carries no such proof and is refused."
        )
        st.stop()
        # st.stop() raises StopException inside a Streamlit run, so control never
        # reaches this line and the store is never opened for a refused caller.
        #
        # It is here because the alternative is a silent fail-open. A bare call
        # to this function - a test, a script runner - has no Streamlit runtime to
        # raise from, and st.stop() is then a no-op, so execution would fall
        # through to `return identity` below and hand back None. Every caller
        # reads None as "authentication is switched off" and would open the
        # store and render the dashboard for the caller just refused. Failing
        # loudly is the correct outcome if that ever happens.
        raise RuntimeError(
            "refused an unauthenticated dashboard request: st.stop() did not halt"
        )

    _LOG.info("dashboard access granted: user=%s role=%s via %s",
              identity.username, identity.role, identity.source)
    return identity


def main(args: argparse.Namespace) -> int:
    """Render the dashboard."""
    # module-level st/go are bound at import time by the optional-extra probe;
    # _require_extras() exists only to give a clear error when they are missing.

    config: Config = load_config(args.config, use_user_config=not args.no_user_config)

    # Before the store, before any query. A refused caller must not be able to
    # read a single row, and opening the database first would make that ordering
    # an accident rather than a property.
    identity = _require_auth(config)

    store = Store(config.paths.database_path)
    try:
        return _render(store, identity)
    finally:
        # Streamlit re-executes the script on every rerun, so a bare close at
        # the end of the happy path leaks a WAL connection whenever a widget
        # raises partway through rendering.
        store.close()


def require_write(identity: Any | None) -> bool:
    """Return whether ``identity`` may perform a write action, else explain why.

    The single place the read/write split is enforced. Provided now, before any
    write path exists, because the failure mode of inventing a per-feature check
    is that one feature quietly ships without one.

    ``None`` means authentication is switched off, in which case the dashboard
    is bound to loopback and there is no identity to check -- acting is
    permitted because refusing would lock out a single-operator desktop node
    with nothing to authenticate against.

    Parameters
    ----------
    identity
        The caller, or ``None`` when auth is disabled.

    Returns
    -------
    bool
        Whether the write may proceed.
    """
    if identity is None:
        return True
    if identity.can_write:
        return True
    safe = _esc(identity.username)
    st.error(
        f"Your dashboard account ({safe}) has read-only access. "
        "Ask for the `write` role, or run `est auth set-role "
        f"{safe} --role write` yourself if you administer this node."
    )
    return False


def _render(store: Store, identity: Any | None = None) -> int:
    """Render every tab against an open ``store``.

    Split out of :func:`main` so the connection lifetime is owned by exactly
    one ``try``/``finally`` rather than depending on the render reaching the
    last line.
    """
    st.set_page_config(
        page_title="Emperor Space Tracker",
        page_icon=":penguin:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject futuristic theme
    st.markdown(_THEME_CSS, unsafe_allow_html=True)

    st.title("EMPEROR SPACE TRACKER")
    st.caption(
        "Antarctic fast-ice, colony and space-weather conditions · "
        f"store {store.path} · "
        f'<span style="color:#00ff88;font-family:Share Tech Mono,monospace;">'
        f"system nominal</span>",
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.markdown(
        '<div style="font-family:Orbitron,sans-serif;font-size:10px;color:#00d4ff;letter-spacing:2px;margin-bottom:12px;">'
        "CONTROLS · TELEMETRY WINDOW</div>",
        unsafe_allow_html=True,
    )

    window_label = st.sidebar.selectbox(
        "TELEMETRY WINDOW",
        options=list(_WINDOWS),
        index=list(_WINDOWS).index("24 hours"),
        label_visibility="collapsed",
    )
    hours = _WINDOWS[window_label]
    st.sidebar.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#555577;margin-top:4px;">'
        f"showing last {hours} hours · {window_label}</div>",
        unsafe_allow_html=True,
    )
    if identity is not None:
        role_colour = "#00ff88" if identity.can_write else "#00d4ff"
        st.sidebar.markdown(
            f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border-default);'
            f'font-family:Share Tech Mono,monospace;font-size:10px;">'
            f'<span style="color:#555577;">SIGNED IN AS</span><br>'
            f'<span style="color:{role_colour};">{_esc(identity.username)}</span>'
            f'<span style="color:#555577;"> · {_esc(identity.role)} · '
            f"{_esc(identity.source)}</span></div>",
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    st.sidebar.markdown(
        '<div style="font-family:Orbitron,sans-serif;font-size:10px;color:#00d4ff;letter-spacing:2px;">'
        "ARCHIVE STATS</div>",
        unsafe_allow_html=True,
    )
    stats = store.stats()
    for s in stats:
        if s.rows:
            st.sidebar.markdown(
                f'<div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#8888aa;">'
                f"{s.table:<16} {s.rows:>8,}</div>",
                unsafe_allow_html=True,
            )
    st.sidebar.markdown(
        f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border-default);font-family:Share Tech Mono,monospace;font-size:11px;color:#00d4ff;">'
        f"TOTAL: {sum(s.rows for s in stats):,} rows</div>",
        unsafe_allow_html=True,
    )

    # Tabs
    overview, weather, ice, colonies, alerts = st.tabs(
        [
            "overview",
            "space weather",
            "fast ice",
            "colonies",
            "alerts",
        ]
    )

    with overview:
        _tab_overview(st, store)
    with weather:
        _tab_space_weather(st, go, store, hours)
    with ice:
        _tab_fast_ice(st, go, store)
    with colonies:
        _tab_colonies(st, go, store)
    with alerts:
        _tab_alerts(st, store)

    return 0


# The telemetry windows the sidebar offers. Declared before `main` uses it.
_WINDOWS: Final[dict[str, int]] = {
    "6 hours": 6,
    "24 hours": 24,
    "7 days": 168,
    "30 days": 720,
}

__all__ = ["_WINDOWS", "main"]


def _streamlit_argv() -> argparse.Namespace:
    """Build the namespace main expects when Streamlit runs this file.

    ``est dashboard`` passes the bind address and port to ``streamlit run`` as
    server options, so they are the process's own concern by the time this
    script body executes and are not read back from here.
    """
    import os

    config_path = os.environ.get("EST_CONFIG") or None
    return argparse.Namespace(
        config=config_path,
        no_user_config=bool(os.environ.get("EST_NO_USER_CONFIG")),
    )


if __name__ == "__main__":
    main(_streamlit_argv())
