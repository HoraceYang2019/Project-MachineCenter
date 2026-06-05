"""
Standalone Dash dashboard for selecting STL and marking grid cells by click.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import dash
from dash import html, dcc, Input, Output, State, ALL
import plotly.graph_objs as go
import trimesh
import numpy as np
from theoryRA import MATERIALS as THEORY_MATERIALS, calculate_theory_ra


DATA_DIR = Path(__file__).resolve().parent / "data"

BASE_TEXT_STYLE = {"fontFamily": "Segoe UI, Arial, sans-serif", "color": "#1f2937"}
TABLE_STYLE = {
    "width": "100%",
    "borderCollapse": "collapse",
    "fontSize": "12px",
    "lineHeight": "1.4",
    "tableLayout": "fixed",
}
TH_STYLE = {
    "backgroundColor": "#eef2f7",
    "textAlign": "left",
    "fontWeight": "600",
    "padding": "6px 8px",
    "borderBottom": "1px solid #d3dbe6",
}
TD_STYLE = {
    "padding": "6px 8px",
    "borderBottom": "1px solid #e5e9f2",
    "verticalAlign": "top",
    "whiteSpace": "normal",
    "wordBreak": "break-word",
}
SECTION_TITLE_STYLE = {"margin": "2px 0 6px 0", "fontSize": "13px", "fontWeight": "600"}
CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "8px",
    "borderRadius": "8px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
}
THEORY_MATERIAL_OPTIONS = [{"label": name, "value": name} for name in THEORY_MATERIALS]


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


def list_stl_files() -> list[dict]:
    if not DATA_DIR.exists():
        return []
    options = []
    for p in sorted(DATA_DIR.glob("*.stl")):
        options.append({"label": p.name, "value": str(p)})
    return options


def load_stl_mesh(stl_path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    xy_min = np.min(vertices[:, :2], axis=0)
    xy_max = np.max(vertices[:, :2], axis=0)
    xy_center = 0.5 * (xy_min + xy_max)
    vertices = vertices.copy()
    vertices[:, 0] -= xy_center[0]
    vertices[:, 1] -= xy_center[1]
    return vertices, faces


def compute_bounds(vertices: np.ndarray) -> Bounds:
    return Bounds(
        xmin=float(np.min(vertices[:, 0])),
        xmax=float(np.max(vertices[:, 0])),
        ymin=float(np.min(vertices[:, 1])),
        ymax=float(np.max(vertices[:, 1])),
        zmin=float(np.min(vertices[:, 2])),
        zmax=float(np.max(vertices[:, 2])),
    )


def grid_size(bounds: Bounds, dx: float, dy: float) -> tuple[int, int]:
    nx = int(math.ceil((bounds.xmax - bounds.xmin) / dx))
    ny = int(math.ceil((bounds.ymax - bounds.ymin) / dy))
    return nx, ny


def cell_from_xy(x: float, y: float, bounds: Bounds, dx: float, dy: float) -> tuple[int, int] | None:
    if x < bounds.xmin or x > bounds.xmax or y < bounds.ymin or y > bounds.ymax:
        return None
    if x == bounds.xmax:
        i = int(math.ceil((bounds.xmax - bounds.xmin) / dx)) - 1
    else:
        i = int(math.floor((x - bounds.xmin) / dx))
    if y == bounds.ymax:
        j = int(math.ceil((bounds.ymax - bounds.ymin) / dy)) - 1
    else:
        j = int(math.floor((y - bounds.ymin) / dy))
    return i, j


def cell_box_mesh(i: int, j: int, bounds: Bounds, dx: float, dy: float, dz: float) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
    x0 = bounds.xmin + i * dx
    x1 = bounds.xmin + (i + 1) * dx
    y0 = bounds.ymin + j * dy
    y1 = bounds.ymin + (j + 1) * dy
    z1 = bounds.zmax
    z0 = bounds.zmax - dz
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    i_idx = [f[0] for f in faces]
    j_idx = [f[1] for f in faces]
    k_idx = [f[2] for f in faces]
    return xs, ys, zs, i_idx, j_idx, k_idx


def build_cell_table(
    cells: list[dict],
    bounds: Bounds,
    dx: float,
    dy: float,
    dz: float,
    selected_index: int | None,
    machining_rows: list[dict],
) -> html.Table:
    def format_optional(value) -> str:
        if value is None or value == "":
            return "-"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    header = html.Thead(
        html.Tr([
            html.Th("Section ID", style=TH_STYLE),
            html.Th("cell(i,j)", style=TH_STYLE),
            html.Th("X range", style=TH_STYLE),
            html.Th("Y range", style=TH_STYLE),
            html.Th("Z range", style=TH_STYLE),
            html.Th("Ra_theoretical", style=TH_STYLE),
            html.Th("Ra_predicted", style=TH_STYLE),
            html.Th("Ra_measured", style=TH_STYLE),
            html.Th("Ra_tolerance", style=TH_STYLE),
        ])
    )
    rows = []
    for idx, cell in enumerate(cells, start=1):
        i = cell.get("i")
        j = cell.get("j")
        is_selected = selected_index == idx
        row_data = machining_rows[idx - 1] if idx - 1 < len(machining_rows) else None
        section_label = "-" if row_data is None else row_data["selection_id"]
        theoretical_label = "-" if row_data is None else format_optional(row_data["ra_theoretical_um"])
        predicted_label = "-" if row_data is None else format_optional(row_data["ra_predicted_um"])
        measured_label = "-" if row_data is None else format_optional(row_data["ra_measured_um"])
        tolerance_label = "-" if row_data is None else format_optional(row_data["ra_tolerance_um"])
        if i is None or j is None:
            cell_label = "(-,-)"
            x_range = "-"
            y_range = "-"
            z_range = "-"
        else:
            cell_label = f"({i},{j})"
            x0 = bounds.xmin + i * dx
            x1 = bounds.xmin + (i + 1) * dx
            y0 = bounds.ymin + j * dy
            y1 = bounds.ymin + (j + 1) * dy
            z0 = bounds.zmax - dz
            z1 = bounds.zmax
            x_range = f"[{x0:.4f}, {x1:.4f}]"
            y_range = f"[{y0:.4f}, {y1:.4f}]"
            z_range = f"[{z0:.4f}, {z1:.4f}]"
        rows.append(
            html.Tr(
                [
                    html.Td(section_label, style=TD_STYLE),
                    html.Td(cell_label, style=TD_STYLE),
                    html.Td(x_range, style=TD_STYLE),
                    html.Td(y_range, style=TD_STYLE),
                    html.Td(z_range, style=TD_STYLE),
                    html.Td(theoretical_label, style=TD_STYLE),
                    html.Td(predicted_label, style=TD_STYLE),
                    html.Td(measured_label, style=TD_STYLE),
                    html.Td(tolerance_label, style=TD_STYLE),
                ],
                style={"backgroundColor": "#eaf2ff"} if is_selected else {},
            )
        )
    return html.Table([header, html.Tbody(rows)], style=TABLE_STYLE)



def build_theory_ra_result(result: dict | None, error: str | None = None) -> html.Div:
    if error:
        return html.Div(error, style={"fontSize": "12px", "color": "#b91c1c", "marginTop": "6px"})
    if result is None:
        return html.Div(
            "Fill all required machining parameters to compute Theory Ra.",
            style={"fontSize": "12px", "color": "#6b7280", "marginTop": "6px"},
        )

    value_rows = [
        ("Material", result["material"]),
        ("h (um)", f"{result['h_um']:.3f}"),
        ("delta (um)", f"{result['delta_um']:.3f}"),
        ("delta_h (um)", f"{result['delta_h_um']:.3f}"),
        ("R0 (um)", f"{result['R0_um']:.3f}"),
        ("Ra_theoretical (um)", f"{result['Ra_um']:.3f}"),
    ]
    rows = [
        html.Tr([html.Td(label, style=TD_STYLE), html.Td(value, style=TD_STYLE)])
        for label, value in value_rows
    ]
    return html.Table(
        [
            html.Thead(html.Tr([html.Th("Item", style=TH_STYLE), html.Th("Value", style=TH_STYLE)])),
            html.Tbody(rows),
        ],
        style=TABLE_STYLE,
    )


def empty_figure() -> go.Figure:
    fig = go.Figure(data=[go.Scatter3d(
        x=[], y=[], z=[],
        mode="markers",
        marker=dict(size=1, color="rgba(0,0,0,0)")
    )])
    fig.update_layout(margin=dict(l=8, r=8, t=30, b=8))
    return fig


def mock_machining_selections() -> list[dict]:
    return [
        {
            "selection_id": "Section 1",
            "nc_start": 1204,
            "nc_end": 1212,
            "nc_content": "G1 X24.800 Y10.300 Z-0.200 F820",
            "ra_theoretical_um": 1.6,
            "ra_predicted_um": 1.9,
            "ra_measured_um": 2.1,
            "ra_tolerance_um": 2.5,
            "sth_torque_mean": 10.8,
            "sth_torque_max": 13.2,
        },
        {
            "selection_id": "Section 2",
            "nc_start": 1186,
            "nc_end": 1192,
            "nc_content": "G1 X18.200 Y6.700 Z-0.150 F780",
            "ra_theoretical_um": 1.4,
            "ra_predicted_um": 1.8,
            "ra_measured_um": 1.9,
            "ra_tolerance_um": 2.3,
            "sth_torque_mean": 9.6,
            "sth_torque_max": 12.1,
        },
        {
            "selection_id": "Section 3",
            "nc_start": 1225,
            "nc_end": 1231,
            "nc_content": "G1 X29.400 Y14.900 Z-0.250 F900",
            "ra_theoretical_um": 1.9,
            "ra_predicted_um": 2.3,
            "ra_measured_um": 2.5,
            "ra_tolerance_um": 2.8,
            "sth_torque_mean": 12.4,
            "sth_torque_max": 15.0,
        },
    ]


def mock_strategy_table() -> list[dict]:
    return [
        {
            "strategy": "Quality-oriented",
            "spindle_rpm": 9800,
            "feed_mm_min": 680,
            "delta_time_percent": 18,
            "ra_theoretical_mean_um": 1.2,
            "ra_factor_theoretical": 0.85,
            "ra_factor_actual": 0.9,
            "nc_adjustment": "Reduce step-over, add finish pass",
        },
        {
            "strategy": "Efficiency-oriented (MMR)",
            "spindle_rpm": 12000,
            "feed_mm_min": 1350,
            "delta_time_percent": -15,
            "ra_theoretical_mean_um": 2.4,
            "ra_factor_theoretical": 1.25,
            "ra_factor_actual": 1.2,
            "nc_adjustment": "Increase feed, skip finish pass",
        },
        {
            "strategy": "Wear-suppression",
            "spindle_rpm": 9000,
            "feed_mm_min": 820,
            "delta_time_percent": 8,
            "ra_theoretical_mean_um": 1.7,
            "ra_factor_theoretical": 0.95,
            "ra_factor_actual": 1.0,
            "nc_adjustment": "Lower engagement, reduce depth per pass",
        },
        {
            "strategy": "Balanced",
            "spindle_rpm": 10500,
            "feed_mm_min": 980,
            "delta_time_percent": 0,
            "ra_theoretical_mean_um": 1.6,
            "ra_factor_theoretical": 1.0,
            "ra_factor_actual": 1.0,
            "nc_adjustment": "Maintain current toolpath with mild feed trim",
        },
    ]


def apply_tolerance(rows: list[dict], tolerance_map: dict[str, float] | None) -> list[dict]:
    if not tolerance_map:
        return [dict(row) for row in rows]
    updated = []
    for row in rows:
        copy_row = dict(row)
        override = tolerance_map.get(row["selection_id"])
        if override is not None:
            copy_row["ra_tolerance_um"] = override
        updated.append(copy_row)
    return updated


def renumber_sections(rows: list[dict]) -> list[dict]:
    renumbered = []
    for idx, row in enumerate(rows, start=1):
        copy_row = dict(row)
        copy_row["selection_id"] = f"Section {idx}"
        renumbered.append(copy_row)
    return renumbered


def compute_machining_sequence(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    ordered = sorted(rows, key=lambda r: r["nc_start"])
    seq_map = {row["selection_id"]: idx for idx, row in enumerate(ordered, start=1)}
    return ordered, seq_map


def build_machining_area_table(rows: list[dict], seq_map: dict[str, int]) -> html.Table:
    header = html.Thead(
        html.Tr([
            html.Th("Section ID", style=TH_STYLE),
            html.Th("NC_No.", style=TH_STYLE),
            html.Th("Machining_order", style=TH_STYLE),
            html.Th("NC_content", style=TH_STYLE),
            html.Th("Torque_mean", style=TH_STYLE),
            html.Th("Torque_max", style=TH_STYLE),
            html.Th("Ra_theoretical", style=TH_STYLE),
            html.Th("Ra_predicted", style=TH_STYLE),
            html.Th("Ra_measured", style=TH_STYLE),
            html.Th("Ra_tolerance", style=TH_STYLE),
        ])
    )
    body_rows = []
    for row in rows:
        nc_range = f"{row['nc_start']}-{row['nc_end']}"
        order_value = seq_map.get(row["selection_id"], "-")
        measured = row["ra_measured_um"]
        measured_label = "-" if measured is None else f"{measured:.2f}"
        body_rows.append(
            html.Tr([
                html.Td(row["selection_id"], style=TD_STYLE),
                html.Td(nc_range, style=TD_STYLE),
                html.Td(str(order_value), style=TD_STYLE),
                html.Td(row["nc_content"], style=TD_STYLE),
                html.Td(f"{row['sth_torque_mean']:.2f}", style=TD_STYLE),
                html.Td(f"{row['sth_torque_max']:.2f}", style=TD_STYLE),
                html.Td(f"{row['ra_theoretical_um']:.2f}", style=TD_STYLE),
                html.Td(f"{row['ra_predicted_um']:.2f}", style=TD_STYLE),
                html.Td(measured_label, style=TD_STYLE),
                html.Td(f"{row['ra_tolerance_um']:.2f}", style=TD_STYLE),
            ])
        )
    return html.Table([header, html.Tbody(body_rows)], style=TABLE_STYLE)


def build_ra_evolution_curve(ordered_rows: list[dict]) -> go.Figure:
    x_vals = list(range(1, len(ordered_rows) + 1))
    predicted_vals = [row["ra_predicted_um"] for row in ordered_rows]
    measured_vals = [
        row["ra_measured_um"] if row["ra_measured_um"] is not None else np.nan
        for row in ordered_rows
    ]
    tolerance_vals = [row["ra_tolerance_um"] for row in ordered_rows]
    labels = [row["selection_id"] for row in ordered_rows]
    y_axis_max = max([v for v in tolerance_vals if v is not None] + predicted_vals)
    y_axis_min = min([v for v in predicted_vals if v is not None] + [0.0])
    y_pad = (y_axis_max - y_axis_min) * 0.18 if y_axis_max > y_axis_min else 0.5
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=predicted_vals,
                mode="lines+markers+text",
                text=labels,
                textposition="bottom center",
                line=dict(color="#ff7a18", width=2),
                marker=dict(size=8),
                name="Predicted",
                cliponaxis=False,
            ),
            go.Scatter(
                x=x_vals,
                y=measured_vals,
                mode="lines+markers",
                line=dict(color="#2b6cb0", width=2),
                marker=dict(size=7),
                name="Measured",
            ),
            go.Scatter(
                x=x_vals,
                y=tolerance_vals,
                mode="lines",
                line=dict(color="#d64541", width=2, dash="dash"),
                name="Tolerance limit",
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Ra Evolution vs Machining Sequence"),
        xaxis_title="Machining Sequence",
        yaxis_title="Ra (um)",
        yaxis=dict(range=[y_axis_min, y_axis_max + y_pad]),
        margin=dict(l=30, r=10, t=40, b=30),
        height=260,
        autosize=False,
    )
    return fig


def build_strategy_table(rows: list[dict]) -> html.Table:
    header = html.Thead(
        html.Tr([
            html.Th("Strategy", style=TH_STYLE),
            html.Th("Spindle_rpm", style=TH_STYLE),
            html.Th("Feed_mm_min", style=TH_STYLE),
            html.Th("DeltaTime_percent", style=TH_STYLE),
            html.Th("Ra_theoretical_mean", style=TH_STYLE),
        ])
    )
    body_rows = []
    for row in rows:
        body_rows.append(
            html.Tr([
                html.Td(row["strategy"], style=TD_STYLE),
                html.Td(str(row["spindle_rpm"]), style=TD_STYLE),
                html.Td(str(row["feed_mm_min"]), style=TD_STYLE),
                html.Td(f"{row['delta_time_percent']}%", style=TD_STYLE),
                html.Td(f"{row['ra_theoretical_mean_um']:.2f}", style=TD_STYLE),
            ])
        )
    return html.Table([header, html.Tbody(body_rows)], style=TABLE_STYLE)


def build_strategy_ra_curve(strategy: dict, ordered_rows: list[dict]) -> go.Figure:
    x_vals = list(range(1, len(ordered_rows) + 1))
    base_theoretical = [row["ra_theoretical_um"] for row in ordered_rows]
    base_actual = [
        row["ra_measured_um"] if row["ra_measured_um"] is not None else row["ra_predicted_um"]
        for row in ordered_rows
    ]
    tolerance_vals = [row["ra_tolerance_um"] for row in ordered_rows]
    labels = [row["selection_id"] for row in ordered_rows]
    theoretical_vals = [v * strategy["ra_factor_theoretical"] for v in base_theoretical]
    actual_vals = [v * strategy["ra_factor_actual"] for v in base_actual]
    y_axis_max = max(tolerance_vals + theoretical_vals + actual_vals)
    y_axis_min = min(theoretical_vals + actual_vals + [0.0])
    y_pad = (y_axis_max - y_axis_min) * 0.18 if y_axis_max > y_axis_min else 0.5
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=theoretical_vals,
                mode="lines+markers+text",
                text=labels,
                textposition="bottom center",
                line=dict(color="#6c4eb6", width=2),
                marker=dict(size=7),
                name="Theoretical",
                cliponaxis=False,
            ),
            go.Scatter(
                x=x_vals,
                y=tolerance_vals,
                mode="lines",
                line=dict(color="#d64541", width=2, dash="dash"),
                name="Tolerance limit",
            ),
        ]
    )
    fig.update_layout(
        title=dict(text="Strategy Ra Curve"),
        xaxis_title="Machining Sequence",
        yaxis_title="Ra (um)",
        yaxis=dict(range=[y_axis_min, y_axis_max + y_pad]),
        margin=dict(l=30, r=10, t=40, b=30),
        height=260,
        autosize=False,
    )
    return fig


def build_strategy_detail(strategy: dict) -> html.Div:
    return html.Table(
        [
            html.Thead(
                html.Tr([html.Th("Item", style=TH_STYLE), html.Th("Value", style=TH_STYLE)])
            ),
            html.Tbody([
                html.Tr([html.Td("Strategy", style=TD_STYLE), html.Td(strategy["strategy"], style=TD_STYLE)]),
                html.Tr([html.Td("Spindle_rpm", style=TD_STYLE), html.Td(str(strategy["spindle_rpm"]), style=TD_STYLE)]),
                html.Tr([html.Td("Feed_mm_min", style=TD_STYLE), html.Td(str(strategy["feed_mm_min"]), style=TD_STYLE)]),
                html.Tr([html.Td("DeltaTime_percent", style=TD_STYLE), html.Td(f"{strategy['delta_time_percent']}%", style=TD_STYLE)]),
                html.Tr([html.Td("Ra_theoretical_mean", style=TD_STYLE), html.Td(f"{strategy['ra_theoretical_mean_um']:.2f}", style=TD_STYLE)]),
                html.Tr([html.Td("NC_adjustment", style=TD_STYLE), html.Td(strategy["nc_adjustment"], style=TD_STYLE)]),
            ]),
        ],
        style=TABLE_STYLE,
    )


def build_section_plan_table(rows: list[dict]) -> html.Table:
    header = html.Thead(
        html.Tr([
            html.Th("Section ID", style=TH_STYLE),
            html.Th("Tolerance", style=TH_STYLE),
            html.Th("Measured", style=TH_STYLE),
            html.Th("Plan Status", style=TH_STYLE),
            html.Th("Delete", style=TH_STYLE),
        ])
    )
    body_rows = []
    for row in rows:
        section_id = row["selection_id"]
        body_rows.append(
            html.Tr([
                html.Td(
                    html.Button(
                        f"Edit {section_id}",
                        id={"type": "section-edit", "index": section_id},
                        n_clicks=0,
                    ),
                    style=TD_STYLE,
                ),
                html.Td(
                    dcc.Input(
                        id={"type": "section-tolerance", "index": section_id},
                        type="number",
                        min=0,
                        step=0.001,
                        value=row["ra_tolerance_um"],
                        style={"width": "100%"},
                        persistence=True,
                    ),
                    style=TD_STYLE,
                ),
                html.Td(
                    dcc.Input(
                        id={"type": "section-measured", "index": section_id},
                        type="number",
                        min=0,
                        step=0.001,
                        value=row.get("ra_measured_um"),
                        style={"width": "100%"},
                        persistence=True,
                    ),
                    style=TD_STYLE,
                ),
                html.Td(
                    html.Span(id={"type": "section-status", "index": section_id}),
                    style=TD_STYLE,
                ),
                html.Td(
                    html.Button(
                        "Delete",
                        id={"type": "section-delete", "index": section_id},
                        n_clicks=0,
                    ),
                    style=TD_STYLE,
                ),
            ])
        )
    return html.Table([header, html.Tbody(body_rows)], style=TABLE_STYLE)


def build_left_column() -> html.Div:
    return html.Div(
        style={
            "flex": "1 1 50%",
            "display": "grid",
            "gridTemplateRows": "8fr 2fr",
            "gap": "8px",
            "minWidth": "0",
            "height": "100%",
            "overflow": "hidden",
        },
        children=[
            html.Div(
                style={"position": "relative", "minHeight": "220px"},
                children=[
                    dcc.Graph(
                        id="stl-graph",
                        style={"minHeight": "220px"},
                        config={"responsive": True},
                    ),
                    html.Div(
                        style={
                            "position": "absolute",
                            "top": "8px",
                            "left": "8px",
                            "width": "220px",
                            "zIndex": 5,
                        },
                        children=[
                            dcc.Dropdown(
                                id="stl-dropdown",
                                options=list_stl_files(),
                                placeholder="Select STL",
                                persistence=True,
                            ),
                            html.Div("Part View", style={"marginTop": "6px", "fontSize": "12px", "fontWeight": "600"}),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="cell-table",
                style={
                    **CARD_STYLE,
                    "padding": "6px",
                    "overflowY": "auto",
                },
            ),
        ],
    )


def build_dashboard_left_column() -> html.Div:
    return html.Div(
        style={
            "flex": "1 1 50%",
            "display": "grid",
            "gridTemplateRows": "8fr 2fr",
            "minWidth": "0",
            "height": "100%",
            "overflow": "hidden",
            "gap": "8px",
        },
        children=[
            html.Div(
                style={"position": "relative", "minHeight": "220px"},
                children=[
                    dcc.Graph(
                        id="stl-graph",
                        style={"minHeight": "220px"},
                        config={"responsive": True},
                    ),
                    html.Div(
                        style={
                            "position": "absolute",
                            "top": "8px",
                            "left": "8px",
                            "width": "220px",
                            "zIndex": 5,
                        },
                        children=[
                            dcc.Dropdown(
                                id="stl-dropdown",
                                options=list_stl_files(),
                                placeholder="Select STL",
                                persistence=True,
                            ),
                            html.Div("Part View", style={"marginTop": "6px", "fontSize": "12px", "fontWeight": "600"}),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="cell-table",
                style={
                    **CARD_STYLE,
                    "padding": "6px",
                    "overflowY": "auto",
                    "overflowX": "hidden",
                },
            ),
            html.Div(
                style={"display": "none"},
                children=[
                    dcc.Input(id="cell-dx-input", type="number", value=None),
                    dcc.Input(id="cell-dy-input", type="number", value=None),
                    dcc.Input(id="cell-dz-input", type="number", value=None),
                    html.Button("Apply", id="cell-apply-grid"),
                ],
            ),
        ],
    )

def build_dashboard_right_column() -> html.Div:
    return html.Div(
        style={
            "flex": "1 1 50%",
            **CARD_STYLE,
            "padding": "10px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "minWidth": "0",
            "overflow": "hidden",
            "height": "100%",
        },
        children=[
            html.Div(
                style={"overflowY": "auto", "overflowX": "hidden", "height": "100%"},
                children=[
                    html.Div([
                        html.H4("Suggested Parameter Table (What-If Strategies)", style=SECTION_TITLE_STYLE),
                        build_strategy_table(strategy_rows),
                    ]),
                    html.Div([
                        html.H4("Strategy Ra Curve", style=SECTION_TITLE_STYLE),
                        dcc.Dropdown(
                            id="strategy-dropdown",
                            options=[{"label": row["strategy"], "value": row["strategy"]} for row in strategy_rows],
                            value=strategy_default["strategy"],
                            clearable=False,
                        ),
                        dcc.Graph(
                            id="strategy-ra-graph",
                            figure=strategy_ra_figure,
                            config={"displayModeBar": False, "scrollZoom": False, "responsive": False},
                            style={"height": "260px", "width": "100%"},
                        ),
                        html.Div(id="strategy-detail", children=build_strategy_detail(strategy_default)),
                    ]),
                    html.Div([
                        html.H4("Machining Area List (Section Table)", style=SECTION_TITLE_STYLE),
                        html.Div(id="machining-area-table", children=build_machining_area_table(ordered_rows, sequence_map)),
                    ]),
                    html.Div([
                        html.H4("Ra Evolution Curve (Actual Machining Sequence)", style=SECTION_TITLE_STYLE),
                        dcc.Graph(
                            id="ra-evolution-graph",
                            figure=ra_curve_figure,
                            config={"displayModeBar": False, "scrollZoom": False, "responsive": False},
                            style={"height": "260px", "width": "100%"},
                        ),
                    ]),
                ],
            ),
        ],
    )


def build_planner_right_column() -> html.Div:
    return html.Div(
        style={
            "flex": "1 1 50%",
            **CARD_STYLE,
            "padding": "10px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "minWidth": "0",
            "overflow": "hidden",
            "height": "100%",
        },
        children=[
            html.Div(
                style={"overflowY": "auto", "overflowX": "hidden", "height": "100%"},
                children=[
                    dcc.Tabs(
                        id="planner-tabs",
                        value="section-plan",
                        children=[
                            dcc.Tab(
                                label="Section Plan",
                                value="section-plan",
                                children=[
                                    html.Div([
                                        html.H4("Measurement & Tolerance Plan", style=SECTION_TITLE_STYLE),
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "flexWrap": "wrap",
                                                "gap": "8px",
                                                "alignItems": "center",
                                                "marginBottom": "6px",
                                            },
                                            children=[
                                                html.Span("dx"),
                                                dcc.Input(
                                                    id="cell-dx-input",
                                                    type="number",
                                                    value=1.0,
                                                    step=0.1,
                                                    min=0.0001,
                                                    persistence=True,
                                                    style={"width": "90px"},
                                                ),
                                                html.Span("dy"),
                                                dcc.Input(
                                                    id="cell-dy-input",
                                                    type="number",
                                                    value=1.0,
                                                    step=0.1,
                                                    min=0.0001,
                                                    persistence=True,
                                                    style={"width": "90px"},
                                                ),
                                                html.Span("dz"),
                                                dcc.Input(
                                                    id="cell-dz-input",
                                                    type="number",
                                                    value=1.0,
                                                    step=0.1,
                                                    min=0.0001,
                                                    persistence=True,
                                                    style={"width": "90px"},
                                                ),
                                                html.Button("Apply", id="cell-apply-grid"),
                                            ],
                                        ),
                                        html.Div(id="section-plan-table", children=build_section_plan_table(machining_rows)),
                                        html.Div(
                                            style={"marginTop": "8px"},
                                            children=[html.Button("Add Section", id="add-section", n_clicks=0)],
                                        ),
                                        html.Div(
                                            style={"marginTop": "6px"},
                                            children=[
                                                html.Button(
                                                    "Export Sections (JSON)",
                                                    id="export-sections-json",
                                                    n_clicks=0,
                                                )
                                            ],
                                        ),
                                        html.Pre(
                                            id="sections-json-output",
                                            style={
                                                "marginTop": "6px",
                                                "padding": "8px",
                                                "backgroundColor": "#f8fafc",
                                                "border": "1px solid #e2e8f0",
                                                "borderRadius": "6px",
                                                "fontSize": "11px",
                                                "whiteSpace": "pre-wrap",
                                                "wordBreak": "break-word",
                                                "maxHeight": "220px",
                                                "overflowY": "auto",
                                            },
                                        ),
                                    ]),
                                    html.Div(
                                        id="plan-completeness-note",
                                        style={"fontSize": "12px", "color": "#6b7280"},
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Machining Parameters",
                                value="machining-params",
                                children=[
                                    html.Div([
                                        html.H4("Machining Parameters", style=SECTION_TITLE_STYLE),
                                        html.Div(
                                            style={
                                                "display": "grid",
                                                "gridTemplateColumns": "140px 1fr 1fr",
                                                "gap": "8px 10px",
                                                "alignItems": "center",
                                                "fontSize": "12px",
                                            },
                                            children=[
                                                html.Div("Tool ID"),
                                                dcc.Input(id="tool-id", type="text", placeholder="T01", persistence=True),
                                                html.Div(),
                                                html.Div("Tool diameter (mm)"),
                                                dcc.Input(id="tool-diameter", type="number", min=0, step=0.01, persistence=True),
                                                html.Div(),
                                                html.Div("Flutes (z)"),
                                                dcc.Input(id="tool-flutes", type="number", min=1, step=1, persistence=True),
                                                html.Div(),
                                                html.Div("Spindle input"),
                                                dcc.Dropdown(
                                                    id="spindle-mode",
                                                    options=[
                                                        {"label": "RPM", "value": "rpm"},
                                                        {"label": "Cutting speed (Vc)", "value": "vc"},
                                                    ],
                                                    value="rpm",
                                                    clearable=False,
                                                    persistence=True,
                                                ),
                                                html.Div(),
                                                html.Div("Spindle rpm"),
                                                dcc.Input(id="spindle-rpm", type="number", min=0, step=1, persistence=True),
                                                html.Div(),
                                                html.Div("Cutting speed Vc"),
                                                dcc.Input(id="cutting-speed", type="number", min=0, step=0.1, persistence=True),
                                                html.Div("m/min"),
                                                html.Div("Feed input"),
                                                dcc.Dropdown(
                                                    id="feed-mode",
                                                    options=[
                                                        {"label": "Feed (mm/min)", "value": "feed"},
                                                        {"label": "fz (mm/tooth)", "value": "fz"},
                                                    ],
                                                    value="feed",
                                                    clearable=False,
                                                    persistence=True,
                                                ),
                                                html.Div(),
                                                html.Div("Feed (mm/min)"),
                                                dcc.Input(id="feed-mm-min", type="number", min=0, step=1, persistence=True),
                                                html.Div(),
                                                html.Div("fz (mm/tooth)"),
                                                dcc.Input(id="feed-fz", type="number", min=0, step=0.001, persistence=True),
                                                html.Div(),
                                                html.Div("ap"),
                                                dcc.Input(id="ap-value", type="number", min=0, step=0.01, persistence=True),
                                                html.Div("mm"),
                                                html.Div("ae"),
                                                dcc.Input(id="ae-value", type="number", min=0, step=0.01, persistence=True),
                                                html.Div("mm"),
                                                html.Div("Tool nose radius r"),
                                                dcc.Input(
                                                    id="tool-nose-radius",
                                                    type="number",
                                                    min=0,
                                                    step=0.001,
                                                    persistence=True,
                                                    placeholder="Optional",
                                                ),
                                                html.Div("mm (default D/2)"),
                                                html.Div("Approach angle k_r"),
                                                dcc.Input(id="kr-angle", type="number", min=0, max=180, step=0.1, persistence=True),
                                                html.Div("deg"),
                                                html.Div("Cutting force F"),
                                                dcc.Input(id="cutting-force", type="number", min=0, step=1, persistence=True),
                                                html.Div("N"),
                                                html.Div("Coolant"),
                                        dcc.Dropdown(
                                            id="coolant",
                                            options=[
                                                {"label": "On", "value": "On"},
                                                {"label": "Off", "value": "Off"},
                                            ],
                                            value=None,
                                            placeholder="Select",
                                            persistence=True,
                                        ),
                                                html.Div(),
                                                html.Div("Material"),
                                        dcc.Dropdown(
                                            id="material",
                                            options=THEORY_MATERIAL_OPTIONS,
                                            value=None,
                                            placeholder="Select",
                                            persistence=True,
                                        ),
                                                html.Div(),
                                            ],
                                        ),
                                        html.Div(id="param-validation", style={"fontSize": "12px", "color": "#b45309"}),
                                        html.Div(
                                            [
                                                html.H5("Theory RA Result", style={"margin": "8px 0 4px 0", "fontSize": "12px"}),
                                                html.Div(id="theory-ra-result", children=build_theory_ra_result(None)),
                                            ]
                                        ),
                                    ]),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_page_layout(right_column: html.Div) -> html.Div:
    return html.Div(
        style={
            "display": "flex",
            "gap": "10px",
            "flex": "1 1 auto",
            "minHeight": "0",
            "overflowX": "hidden",
            "alignItems": "stretch",
            "height": "100%",
        },
        children=[
            build_left_column(),
            right_column,
        ],
    )


def build_dashboard_layout() -> html.Div:
    return html.Div(
        style={
            "display": "flex",
            "gap": "10px",
            "flex": "1 1 auto",
            "minHeight": "0",
            "overflowX": "hidden",
            "height": "100%",
        },
        children=[
            build_dashboard_left_column(),
            build_dashboard_right_column(),
        ],
    )


def build_planner_controls(visible: bool) -> html.Div:
    style = {
        **CARD_STYLE,
        "padding": "8px",
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "10px",
        "alignItems": "center",
    }
    if not visible:
        style["display"] = "none"
    return html.Div(
        style=style,
        children=[
        ],
    )

def default_cells_for_mesh(bounds: Bounds, dx: float, dy: float) -> list[dict]:
    nx, ny = grid_size(bounds, dx, dy)
    if nx <= 0 or ny <= 0:
        return []
    center_x = max(nx // 2, 0)
    center_y = max(ny // 2, 0)
    inset = max(int(round(nx * 0.25)) - 7, 1)
    left_x = max(min(nx - 1 - inset, nx - 1), 0)
    right_x = min(inset, nx - 1)
    candidates = [
        (left_x, center_y),
        (center_x, center_y),
        (right_x, center_y),
    ]
    cells = []
    for i, j in candidates:
        if 0 <= i < nx and 0 <= j < ny:
            cells.append({"i": i, "j": j})
    return cells


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "STL Cell Dashboard"

machining_rows = mock_machining_selections()
ordered_rows, sequence_map = compute_machining_sequence(machining_rows)
strategy_rows = mock_strategy_table()
ra_curve_figure = build_ra_evolution_curve(ordered_rows)
strategy_default = strategy_rows[0]
strategy_ra_figure = build_strategy_ra_curve(strategy_default, ordered_rows)

app.layout = html.Div(
    style={
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "gap": "0",
        "padding": "0",
        "backgroundColor": "#f4f6fb",
        **BASE_TEXT_STYLE,
    },
    children=[
        html.Div(
            style={"display": "flex", "gap": "10px", "flex": "1 1 auto", "minHeight": "0"},
            children=[
                html.Div(
                    style={
                        **CARD_STYLE,
                        "width": "max-content",
                        "flex": "0 0 auto",
                        "padding": "8px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                        "alignSelf": "stretch",
                    },
                    children=[
                        html.Button(
                            "1️⃣ Planning Sections",
                            id="nav-planner",
                            n_clicks=0,
                        ),
                        html.Button(
                            "2️⃣ What-If Strategies",
                            id="nav-dashboard",
                            n_clicks=0,
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1 1 auto", "display": "flex", "flexDirection": "column", "gap": "8px"},
                    children=[
                        html.Div(id="page-header"),
                        html.Div(
                            id="page-content",
                            style={"flex": "1 1 auto", "minWidth": "0", "overflow": "hidden", "height": "100%"},
                        ),
                    ],
                ),
            ],
        ),
        dcc.Store(id="cell-store", data=[]),
        dcc.Store(id="cell-grid-store", data={"dx": 1.0, "dy": 1.0, "dz": 1.0}),
        dcc.Store(id="cell-selected-index", data=None),
        dcc.Store(id="tolerance-store", data={}),
        dcc.Store(id="params-store", data={"is_complete": False}),
        dcc.Store(id="page-store", data="planner"),
        dcc.Store(id="sections-store", data=machining_rows),
        dcc.Store(id="sections-action", data={}),
    ],
)


@app.callback(Output("page-content", "children"), [Input("page-store", "data")])
def render_page(page_value):
    if page_value == "planner":
        return build_page_layout(build_planner_right_column())
    return build_dashboard_layout()


@app.callback(Output("page-header", "children"), [Input("page-store", "data")])
def render_header(page_value):
    return build_planner_controls(visible=page_value == "planner")


@app.callback(
    Output("page-store", "data"),
    [Input("nav-planner", "n_clicks"), Input("nav-dashboard", "n_clicks")],
    [State("page-store", "data")],
)
def update_page_store(planner_clicks, dashboard_clicks, current):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "nav-dashboard":
        return "dashboard"
    if trigger == "nav-planner":
        return "planner"
    return current


@app.callback(
    [Output("nav-planner", "style"), Output("nav-dashboard", "style")],
    [Input("page-store", "data")],
)
def update_nav_styles(page_value):
    return nav_button_style(page_value == "planner"), nav_button_style(page_value == "dashboard")


@app.callback(
    Output("stl-graph", "figure"),
    [
        Input("stl-dropdown", "value"),
        Input("cell-store", "data"),
        Input("cell-grid-store", "data"),
        Input("tolerance-store", "data"),
        Input("sections-store", "data"),
    ],
)
def update_stl_figure(stl_path, cell_store, grid_store, tolerance_store, sections_store):
    if not stl_path:
        return empty_figure()
    try:
        vertices, faces = load_stl_mesh(stl_path)
        bounds = compute_bounds(vertices)
        dx = float((grid_store or {}).get("dx", 1.0))
        dy = float((grid_store or {}).get("dy", 1.0))
        dz = float((grid_store or {}).get("dz", 1.0))
        if dx <= 0 or dy <= 0 or dz <= 0:
            dx, dy, dz = 1.0, 1.0, 1.0
        nx, ny = grid_size(bounds, dx, dy)

        annotations = []
        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#4682b4",
            opacity=0.50,
            hoverinfo="x+y+z",
            hovertemplate="X %{x:.3f}<br>Y %{y:.3f}<br>Z %{z:.3f}<extra></extra>",
        )])

        effective_rows = apply_tolerance(sections_store or [], tolerance_store)
        for idx, cell in enumerate((cell_store or []), start=1):
            i = cell.get("i")
            j = cell.get("j")
            if i is None or j is None:
                continue
            if i < 0 or i >= nx or j < 0 or j >= ny:
                continue
            xs, ys, zs, ii, jj, kk = cell_box_mesh(i, j, bounds, dx, dy, dz)
            x0 = bounds.xmin + i * dx
            x1 = bounds.xmin + (i + 1) * dx
            y0 = bounds.ymin + j * dy
            y1 = bounds.ymin + (j + 1) * dy
            z0 = bounds.zmax - dz
            z1 = bounds.zmax
            fig.add_trace(go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=ii,
                j=jj,
                k=kk,
                color="#d64541",
                opacity=0.6,
                showscale=False,
                hoverinfo="text",
                hovertext=(
                    f"{effective_rows[idx - 1]['selection_id']}"
                    f"<br>Ra(T) {effective_rows[idx - 1]['ra_theoretical_um']:.2f}"
                    f" | Ra(P) {effective_rows[idx - 1]['ra_predicted_um']:.2f}"
                    f" | Ra(M) {effective_rows[idx - 1]['ra_measured_um']:.2f}"
                    f" | Tol {effective_rows[idx - 1]['ra_tolerance_um']:.2f}"
                    f"<br>X [{x0:.3f}, {x1:.3f}]"
                    f" Y [{y0:.3f}, {y1:.3f}]"
                    f" Z [{z0:.3f}, {z1:.3f}]"
                ) if idx - 1 < len(effective_rows) else "No data",
            ))
            row_data = effective_rows[idx - 1] if idx - 1 < len(effective_rows) else None
            if row_data is not None:
                z_text = bounds.zmax + dz * 5.0
                label = (
                    f"<span style='font-weight:600'>{row_data['selection_id']}</span>"
                    f"<br>Ra(T): {row_data['ra_theoretical_um']:.2f}"
                    f"<br>Ra(P): {row_data['ra_predicted_um']:.2f}"
                    f"<br>Ra(M): {row_data['ra_measured_um']:.2f}"
                )
                annotations.append(
                    dict(
                        x=(x0 + x1) / 2,
                        y=(y0 + y1) / 2,
                        z=z_text,
                        text=label,
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#cbd5e1",
                        borderwidth=1,
                        font=dict(size=12, color="#111827"),
                        align="left",
                    )
                )

        fig.update_layout(
            uirevision=stl_path,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                annotations=annotations,
                aspectmode="data",
                camera=dict(eye=dict(x=2.5, y=1.7, z=1.4))
            ),
            margin=dict(l=8, r=8, t=30, b=8),
            autosize=True,
        )
        return fig
    except Exception:
        return empty_figure()


@app.callback(
    [
        Output("cell-store", "data"),
        Output("cell-grid-store", "data"),
        Output("cell-table", "children"),
        Output("cell-selected-index", "data"),
    ],
    [
        Input("stl-graph", "clickData"),
        Input("cell-apply-grid", "n_clicks"),
        Input("stl-dropdown", "value"),
        Input("tolerance-store", "data"),
        Input("sections-store", "data"),
        Input("sections-action", "data"),
        Input({"type": "section-edit", "index": ALL}, "n_clicks"),
        Input("page-store", "data"),
    ],
    [
        State("cell-dx-input", "value"),
        State("cell-dy-input", "value"),
        State("cell-dz-input", "value"),
        State("cell-store", "data"),
        State("cell-grid-store", "data"),
        State("cell-selected-index", "data"),
    ],
    prevent_initial_call=True,
)
def update_cell_store(click_data, apply_clicks, stl_path, tolerance_store, sections_store, sections_action, edit_clicks, page_store,
                      dx_val, dy_val, dz_val, cell_store, grid_store, selected_index):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    if not stl_path:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    dx = float(dx_val) if dx_val and dx_val > 0 else float((grid_store or {}).get("dx", 1.0))
    dy = float(dy_val) if dy_val and dy_val > 0 else float((grid_store or {}).get("dy", 1.0))
    dz = float(dz_val) if dz_val and dz_val > 0 else float((grid_store or {}).get("dz", 1.0))
    grid_store_new = {"dx": dx, "dy": dy, "dz": dz}

    try:
        vertices, _ = load_stl_mesh(stl_path)
        bounds = compute_bounds(vertices)
    except Exception:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    cells = list(cell_store or [])
    selected = selected_index

    if trigger == "stl-dropdown":
        cells = default_cells_for_mesh(bounds, dx, dy)
        selected = len(cells) if cells else None
    elif trigger == "cell-apply-grid":
        cells = []
        selected = None
    elif trigger == "sections-store":
        pass
    elif trigger == "sections-action":
        deleted_index = (sections_action or {}).get("deleted_index")
        if deleted_index and 1 <= deleted_index <= len(cells):
            cells.pop(deleted_index - 1)
            if selected is not None:
                if selected == deleted_index:
                    selected = None
                elif selected > deleted_index:
                    selected -= 1
    elif trigger.startswith("{") and "section-edit" in trigger:
        try:
            payload = json.loads(trigger)
            target = payload.get("index")
            if sections_store:
                for idx, row in enumerate(sections_store, start=1):
                    if row["selection_id"] == target:
                        selected = idx
                        break
        except Exception:
            selected = selected_index
    elif trigger == "stl-graph":
        if click_data and click_data.get("points"):
            p = click_data["points"][0]
            x = p.get("x")
            y = p.get("y")
            if x is not None and y is not None:
                res = cell_from_xy(x, y, bounds, dx, dy)
                if res is not None:
                    i, j = res
                    if selected is not None:
                        idx = int(selected) - 1
                        if idx >= len(cells):
                            cells.extend([{"i": None, "j": None}] * (idx - len(cells) + 1))
                        if 0 <= idx < len(cells):
                            cells[idx] = {"i": i, "j": j}

    target_len = len(sections_store or [])
    if len(cells) < target_len:
        cells.extend([{"i": None, "j": None}] * (target_len - len(cells)))
    elif len(cells) > target_len:
        cells = cells[:target_len]
    if selected is not None and selected > target_len:
        selected = target_len if target_len > 0 else None

    effective_rows = apply_tolerance(sections_store or [], tolerance_store)
    table = build_cell_table(cells, bounds, dx, dy, dz, selected, effective_rows)
    return cells, grid_store_new, table, selected


@app.callback(
    Output("machining-area-table", "children"),
    [Input("tolerance-store", "data"), Input("sections-store", "data"), Input("page-store", "data")],
)
def update_machining_area_table(tolerance_store, sections_store, page_store):
    rows = sections_store or []
    effective_rows = apply_tolerance(rows, tolerance_store)
    _, seq_map = compute_machining_sequence(rows) if rows else ([], {})
    return build_machining_area_table(effective_rows, seq_map)


@app.callback(
    Output("ra-evolution-graph", "figure"),
    [Input("tolerance-store", "data"), Input("sections-store", "data"), Input("page-store", "data")],
)
def update_ra_evolution_curve(tolerance_store, sections_store, page_store):
    rows = sections_store or []
    ordered, _ = compute_machining_sequence(rows) if rows else ([], {})
    effective_rows = apply_tolerance(ordered, tolerance_store)
    return build_ra_evolution_curve(effective_rows)


@app.callback(
    [Output("strategy-ra-graph", "figure"), Output("strategy-detail", "children")],
    [
        Input("strategy-dropdown", "value"),
        Input("tolerance-store", "data"),
        Input("sections-store", "data"),
        Input("page-store", "data"),
    ],
)
def update_strategy_view(strategy_name, tolerance_store, sections_store, page_store):
    selected = next((row for row in strategy_rows if row["strategy"] == strategy_name), strategy_rows[0])
    rows = sections_store or []
    ordered, _ = compute_machining_sequence(rows) if rows else ([], {})
    effective_rows = apply_tolerance(ordered, tolerance_store)
    return build_strategy_ra_curve(selected, effective_rows), build_strategy_detail(selected)


@app.callback(
    Output("tolerance-store", "data"),
    [Input({"type": "section-tolerance", "index": ALL}, "value")],
    [State({"type": "section-tolerance", "index": ALL}, "id")],
)
def update_tolerance_store(values, ids):
    if not ids:
        return {}
    tolerance_map = {}
    for val, item_id in zip(values, ids):
        if val is None:
            continue
        tolerance_map[item_id["index"]] = float(val)
    return tolerance_map


@app.callback(
    [Output("sections-store", "data"), Output("sections-action", "data")],
    [
        Input("add-section", "n_clicks"),
        Input({"type": "section-delete", "index": ALL}, "n_clicks"),
        Input({"type": "section-measured", "index": ALL}, "value"),
    ],
    [
        State("sections-store", "data"),
        State({"type": "section-delete", "index": ALL}, "id"),
        State({"type": "section-measured", "index": ALL}, "id"),
    ],
)
def update_sections(add_clicks, delete_clicks, measured_values, sections_store, delete_ids, measured_ids):
    sections = list(sections_store or [])
    action = {}
    ctx = dash.callback_context
    if not ctx.triggered:
        sections = renumber_sections(sections)
        return sections, action
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "add-section":
        next_idx = len(sections) + 1
        last_nc = sections[-1]["nc_end"] if sections else 1200
        new_section = {
            "selection_id": f"Section {next_idx}",
            "nc_start": last_nc + 1,
            "nc_end": last_nc + 3,
            "nc_content": "G1 X0.000 Y0.000 Z-0.100 F800",
            "ra_theoretical_um": 1.6,
            "ra_predicted_um": 1.9,
            "ra_measured_um": 2.0,
            "ra_tolerance_um": 2.5,
            "sth_torque_mean": 10.0,
            "sth_torque_max": 12.0,
        }
        sections.append(new_section)
        sections = renumber_sections(sections)
        return sections, action
    elif trigger.startswith("{") and "section-measured" in trigger:
        if measured_ids:
            for val, item_id in zip(measured_values or [], measured_ids):
                target = item_id.get("index")
                for row in sections:
                    if row["selection_id"] == target:
                        row["ra_measured_um"] = None if val is None else float(val)
                        break
        sections = renumber_sections(sections)
        return sections, action
    elif trigger.startswith("{") and "section-delete" in trigger:
        try:
            payload = json.loads(trigger)
            target = payload.get("index")
            click_value = None
            deleted_index = None
            for idx, item_id in enumerate(delete_ids or []):
                if item_id.get("index") == target:
                    if delete_clicks and idx < len(delete_clicks):
                        click_value = delete_clicks[idx]
                    break
            if click_value:
                for idx, row in enumerate(sections, start=1):
                    if row["selection_id"] == target:
                        deleted_index = idx
                        break
                sections = [row for row in sections if row["selection_id"] != target]
                if deleted_index:
                    action = {"deleted_index": deleted_index}
            else:
                return sections, action
        except Exception:
            return sections, action
    sections = renumber_sections(sections)
    return sections, action


@app.callback(
    Output("section-plan-table", "children"),
    [Input("tolerance-store", "data"), Input("sections-store", "data")],
)
def update_section_plan_table(tolerance_store, sections_store):
    effective_rows = apply_tolerance(sections_store or [], tolerance_store)
    return build_section_plan_table(effective_rows)


@app.callback(
    Output("sections-json-output", "children"),
    [Input("export-sections-json", "n_clicks")],
    [
        State("sections-store", "data"),
        State("cell-store", "data"),
        State("cell-grid-store", "data"),
        State("stl-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def export_sections_json(n_clicks, sections_store, cell_store, grid_store, stl_path):
    if not n_clicks:
        return ""
    sections = list(sections_store or [])
    cells = list(cell_store or [])
    if not stl_path:
        payload = [{"section": row.get("selection_id"), "xyz_range": None} for row in sections]
        return json.dumps(payload, ensure_ascii=True, indent=2)

    try:
        vertices, _ = load_stl_mesh(stl_path)
        bounds = compute_bounds(vertices)
    except Exception:
        payload = [{"section": row.get("selection_id"), "xyz_range": None} for row in sections]
        return json.dumps(payload, ensure_ascii=True, indent=2)

    dx = float((grid_store or {}).get("dx", 1.0))
    dy = float((grid_store or {}).get("dy", 1.0))
    dz = float((grid_store or {}).get("dz", 1.0))
    if dx <= 0 or dy <= 0 or dz <= 0:
        dx, dy, dz = 1.0, 1.0, 1.0

    payload = []
    for idx, row in enumerate(sections):
        cell = cells[idx] if idx < len(cells) else {}
        i = cell.get("i")
        j = cell.get("j")
        if i is None or j is None:
            xyz_range = None
        else:
            x0 = bounds.xmin + i * dx
            x1 = bounds.xmin + (i + 1) * dx
            y0 = bounds.ymin + j * dy
            y1 = bounds.ymin + (j + 1) * dy
            z0 = bounds.zmax - dz
            z1 = bounds.zmax
            xyz_range = {
                "x": [round(x0, 3), round(x1, 3)],
                "y": [round(y0, 3), round(y1, 3)],
                "z": [round(z0, 3), round(z1, 3)],
            }
        payload.append({"section": row.get("selection_id"), "xyz_range": xyz_range})

    return json.dumps(payload, ensure_ascii=True, indent=2)


@app.callback(
    [Output({"type": "section-status", "index": ALL}, "children"), Output("plan-completeness-note", "children")],
    [
        Input({"type": "section-tolerance", "index": ALL}, "value"),
        Input("params-store", "data"),
    ],
    [State({"type": "section-status", "index": ALL}, "id")],
)
def update_section_status(tolerances, params_store, ids):
    if not ids:
        return [], ""
    statuses = []
    complete_count = 0
    for tol in tolerances:
        missing = []
        if tol is None:
            missing.append("tolerance")
        if not (params_store or {}).get("is_complete"):
            missing.append("params")
        if missing:
            statuses.append("⚠️")
        else:
            statuses.append("✅")
            complete_count += 1
    note = f"Plan completeness: {complete_count}/{len(ids)} sections complete"
    return statuses, note


@app.callback(
    [
        Output("spindle-rpm", "value"),
        Output("cutting-speed", "value"),
        Output("spindle-rpm", "disabled"),
        Output("cutting-speed", "disabled"),
        Output("feed-mm-min", "value"),
        Output("feed-fz", "value"),
        Output("feed-mm-min", "disabled"),
        Output("feed-fz", "disabled"),
        Output("param-validation", "children"),
        Output("theory-ra-result", "children"),
        Output("params-store", "data"),
    ],
    [
        Input("spindle-mode", "value"),
        Input("spindle-rpm", "value"),
        Input("cutting-speed", "value"),
        Input("tool-diameter", "value"),
        Input("feed-mode", "value"),
        Input("feed-mm-min", "value"),
        Input("feed-fz", "value"),
        Input("tool-flutes", "value"),
        Input("tool-id", "value"),
        Input("ap-value", "value"),
        Input("ae-value", "value"),
        Input("coolant", "value"),
        Input("material", "value"),
        Input("tool-nose-radius", "value"),
        Input("kr-angle", "value"),
        Input("cutting-force", "value"),
    ],
)
def update_params(
    spindle_mode,
    spindle_rpm,
    cutting_speed,
    tool_diameter,
    feed_mode,
    feed_mm_min,
    feed_fz,
    tool_flutes,
    tool_id,
    ap_value,
    ae_value,
    coolant,
    material,
    tool_nose_radius,
    kr_angle,
    cutting_force,
):
    rpm_out = spindle_rpm
    vc_out = cutting_speed
    rpm_disabled = spindle_mode == "vc"
    vc_disabled = spindle_mode == "rpm"

    if tool_diameter and tool_diameter > 0:
        if spindle_mode == "rpm" and spindle_rpm:
            vc_out = (math.pi * tool_diameter * spindle_rpm) / 1000.0
        if spindle_mode == "vc" and cutting_speed:
            rpm_out = (1000.0 * cutting_speed) / (math.pi * tool_diameter)

    rpm_used = rpm_out if rpm_out else None

    feed_out = feed_mm_min
    fz_out = feed_fz
    feed_disabled = feed_mode == "fz"
    fz_disabled = feed_mode == "feed"
    if rpm_used and tool_flutes and tool_flutes > 0:
        if feed_mode == "feed" and feed_mm_min:
            fz_out = feed_mm_min / (rpm_used * tool_flutes)
        if feed_mode == "fz" and feed_fz:
            feed_out = feed_fz * rpm_used * tool_flutes

    warning_messages = []
    if ae_value is not None and tool_diameter is not None and ae_value > tool_diameter:
        warning_messages.append("Warning: ae should be <= tool diameter.")

    tool_nose_radius_out = None
    if tool_nose_radius is not None:
        if tool_nose_radius > 0:
            tool_nose_radius_out = float(tool_nose_radius)
        else:
            warning_messages.append("Warning: tool nose radius r must be > 0.")
    elif tool_diameter and tool_diameter > 0:
        tool_nose_radius_out = float(tool_diameter) / 2.0

    if kr_angle is not None and (kr_angle <= 0 or kr_angle >= 180):
        warning_messages.append("Warning: k_r must be between 0 and 180 degrees.")
    if cutting_force is not None and cutting_force <= 0:
        warning_messages.append("Warning: cutting force must be > 0.")

    required_fields = [
        tool_id,
        tool_diameter,
        tool_flutes,
        ap_value,
        ae_value,
        coolant,
        material,
        tool_nose_radius_out,
        kr_angle,
        cutting_force,
    ]
    spindle_ok = (spindle_mode == "rpm" and rpm_out) or (spindle_mode == "vc" and vc_out)
    feed_ok = (feed_mode == "feed" and feed_out) or (feed_mode == "fz" and fz_out)
    is_complete = all(required_fields) and bool(spindle_ok) and bool(feed_ok)

    theory_result = None
    theory_error = None
    if is_complete:
        try:
            theory_result = calculate_theory_ra(
                workpiece=material,
                r_mm=float(tool_nose_radius_out),
                kr_deg=float(kr_angle),
                fz_mm_per_tooth=float(fz_out),
                cutting_force_n=float(cutting_force),
            )
        except Exception as exc:
            theory_error = f"Theory RA calculation failed: {exc}"
            warning_messages.append(theory_error)
            is_complete = False

    theory_result_view = build_theory_ra_result(theory_result, theory_error)
    warning = " | ".join(warning_messages)

    params_data = {
        "is_complete": is_complete,
        "spindle_rpm": rpm_out,
        "cutting_speed": vc_out,
        "feed_mm_min": feed_out,
        "fz": fz_out,
        "tool_id": tool_id,
        "tool_diameter": tool_diameter,
        "tool_flutes": tool_flutes,
        "ap": ap_value,
        "ae": ae_value,
        "coolant": coolant,
        "material": material,
        "tool_nose_radius": tool_nose_radius_out,
        "kr_angle": kr_angle,
        "cutting_force_n": cutting_force,
        "theory_ra_result": theory_result,
        "ra_theoretical_um": None if theory_result is None else theory_result["Ra_um"],
        "theory_error": theory_error,
    }

    return (
        rpm_out,
        vc_out,
        rpm_disabled,
        vc_disabled,
        feed_out,
        fz_out,
        feed_disabled,
        fz_disabled,
        warning,
        theory_result_view,
        params_data,
    )


def nav_button_style(is_active: bool) -> dict:
    base = {
        "width": "100%",
        "height": "44px",
        "border": "1px solid #d1d5db",
        "borderRadius": "6px",
        "backgroundColor": "white",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "flex-start",
        "cursor": "pointer",
        "fontSize": "12px",
        "color": "#374151",
        "padding": "0 10px",
        "textAlign": "left",
        "whiteSpace": "nowrap",
    }
    if is_active:
        base.update({"backgroundColor": "#e8f0ff", "borderColor": "#93c5fd", "color": "#1d4ed8"})
    return base



if __name__ == "__main__":
    app.run(debug=False)
