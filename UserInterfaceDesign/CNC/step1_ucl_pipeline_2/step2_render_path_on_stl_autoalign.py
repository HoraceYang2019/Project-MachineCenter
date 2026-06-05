"""
Render CNC 3D path from synced_ucl_shifted.csv and overlay it on the STL model.

Defaults:
- Input CSV: path_with_stl/output/synced_ucl_shifted.csv
- STL:       path_with_stl/data/325BTM.STL
- Output:    path_with_stl/output/path_on_stl.html
"""

from pathlib import Path
from typing import Optional
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from stl import mesh as stl_mesh


DEFAULT_DATA_CSV = Path(__file__).resolve().parent / "output" / "aligned_no_ucl.csv"
DEFAULT_REF_CSV = Path(__file__).resolve().parent / "output" / "reference.csv"
DEFAULT_STL = Path(__file__).resolve().parent / "data" / "325BTM.STL"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
COLOR_SCALE = [
    [0.0, "#00008b"],  # deep blue
    [0.5, "#ffa500"],  # orange
    [1.0, "#ff0000"],  # red
]
# Optional hard-coded override. Set to True to always use auto-align-top, False to force legacy recenter,
# or None to keep CLI-controlled behavior.
HARDCODE_AUTO_ALIGN_TOP: Optional[bool] = True


def load_and_merge(data_csv: Path, ref_csv: Path) -> pd.DataFrame:
    data_df = pd.read_csv(data_csv)
    ref_df = pd.read_csv(ref_csv)

    data_needed = ["Offset_X", "Offset_Y", "Offset_Z", "Torque", "BendingX"]
    ref_needed = ["Offset_X", "Offset_Y", "Offset_Z", "Torque_UCL_3", "Torque_UCL_5", "BendingX_UCL_3", "BendingX_UCL_5"]

    miss_data = [c for c in data_needed if c not in data_df.columns]
    miss_ref = [c for c in ref_needed if c not in ref_df.columns]
    if miss_data:
        raise ValueError(f"Data CSV missing columns: {miss_data}")
    if miss_ref:
        raise ValueError(f"Reference CSV missing columns: {miss_ref}")

    merged = data_df.merge(
        ref_df[ref_needed],
        on=["Offset_X", "Offset_Y", "Offset_Z"],
        how="left",
    )
    merged = merged.dropna(subset=["Offset_X", "Offset_Y", "Offset_Z"])
    return merged


def stl_to_mesh3d(stl_path: Path):
    m = stl_mesh.Mesh.from_file(str(stl_path))
    verts = m.vectors.reshape(-1, 3)
    # Deduplicate vertices; round to reduce floating duplicates from STL export.
    verts_rounded = np.round(verts, 5)
    unique, inverse = np.unique(verts_rounded, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return unique, faces


def recenter_stl_to_xy_center_top_z(verts: np.ndarray) -> np.ndarray:
    """
    Shift STL so that:
    - XY origin moves to model center
    - Z=0 is set to the model top
    """
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    xy_center = 0.5 * (vmin[:2] + vmax[:2])
    z_top = vmax[2]
    shift = np.array([xy_center[0], xy_center[1], z_top])
    return verts - shift




def _bbox_xyz(points: np.ndarray):
    """Return (minx,maxx,miny,maxy,minz,maxz,cx,cy) for a (N,3) array."""
    minx = float(np.min(points[:, 0])); maxx = float(np.max(points[:, 0]))
    miny = float(np.min(points[:, 1])); maxy = float(np.max(points[:, 1]))
    minz = float(np.min(points[:, 2])); maxz = float(np.max(points[:, 2]))
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    return minx, maxx, miny, maxy, minz, maxz, cx, cy

# 人類可讀的候選描述，幫助辨識 STL 頂面的哪個角／中心。
CANDIDATE_LABELS = {
    "minx_miny_topz": "minX, minY, topZ (左下角，頂面)",
    "minx_maxy_topz": "minX, maxY, topZ (左上角，頂面)",
    "maxx_miny_topz": "maxX, minY, topZ (右下角，頂面)",
    "maxx_maxy_topz": "maxX, maxY, topZ (右上角，頂面)",
    "center_topz": "center XY, topZ (頂面中心)",
}


def _stl_origin_top_surface(verts: np.ndarray, mode: str) -> np.ndarray:
    """
    Compute STL anchor/origin (in STL coords) for top-surface-only alignment.
    Z is anchored at the STL top surface (maxZ).

    Modes:
      - minx_miny_topz
      - minx_maxy_topz
      - maxx_miny_topz
      - maxx_maxy_topz
      - center_topz
    """
    minx, maxx, miny, maxy, minz, maxz, cx, cy = _bbox_xyz(verts)
    if mode == "minx_miny_topz":
        return np.array([minx, miny, maxz], dtype=float)
    if mode == "minx_maxy_topz":
        return np.array([minx, maxy, maxz], dtype=float)
    if mode == "maxx_miny_topz":
        return np.array([maxx, miny, maxz], dtype=float)
    if mode == "maxx_maxy_topz":
        return np.array([maxx, maxy, maxz], dtype=float)
    if mode == "center_topz":
        return np.array([cx, cy, maxz], dtype=float)
    raise ValueError(f"Unknown mode: {mode}")


def _score_alignment_top_surface(
    cnc_xyz: np.ndarray,
    stl_verts_aligned: np.ndarray,
    eps_xy: float = 0.05,
    eps_z: float = 0.05,
    sample_n: int = 8000,
) -> dict:
    """
    Score how well CNC points fit within aligned STL bounding box.
    Returns dict with score and components. Higher is better.
    """
    if cnc_xyz.size == 0 or stl_verts_aligned.size == 0:
        return {"score": -1e9, "in_ratio": 0.0, "penalty": 1e9, "quad": 0.0}

    # Downsample CNC points for speed (keep distribution)
    n = cnc_xyz.shape[0]
    if n > sample_n:
        idx = np.linspace(0, n - 1, sample_n).astype(int)
        P = cnc_xyz[idx]
    else:
        P = cnc_xyz

    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    minx, maxx, miny, maxy, minz, maxz, cx, cy = _bbox_xyz(stl_verts_aligned)

    in_box = (
        (x >= minx - eps_xy) & (x <= maxx + eps_xy) &
        (y >= miny - eps_xy) & (y <= maxy + eps_xy) &
        (z >= minz - eps_z)  & (z <= maxz + eps_z)
    )
    in_ratio = float(np.mean(in_box))

    # distance outside bounding box (L1) as penalty
    dx = np.maximum(0.0, x - (maxx + eps_xy)) + np.maximum(0.0, (minx - eps_xy) - x)
    dy = np.maximum(0.0, y - (maxy + eps_xy)) + np.maximum(0.0, (miny - eps_xy) - y)
    dz = np.maximum(0.0, z - (maxz + eps_z))  + np.maximum(0.0, (minz - eps_z) - z)
    penalty = float(np.mean(dx + dy + dz))

    # quadrant concentration (auxiliary)
    quad = float(max(
        np.mean((x >= -eps_xy) & (y >= -eps_xy)),
        np.mean((x <=  eps_xy) & (y >= -eps_xy)),
        np.mean((x <=  eps_xy) & (y <=  eps_xy)),
        np.mean((x >= -eps_xy) & (y <=  eps_xy)),
    ))

    # Combine score (tuned for stability; can adjust later)
    score = (1.0 * in_ratio) - (0.5 * penalty) + (0.15 * quad)
    return {"score": float(score), "in_ratio": in_ratio, "penalty": penalty, "quad": quad}


def auto_align_stl_top_surface(
    stl_verts: np.ndarray,
    cnc_xyz: np.ndarray,
    candidates=None,
    eps_xy: float = 0.05,
    eps_z: float = 0.05,
    verbose: bool = True,
):
    """
    Automatically choose an STL-origin mode (top surface only) that best fits CNC points.

    Returns:
      (aligned_verts, best_mode, debug_scores, origin_vector)
    """
    if candidates is None:
        candidates = [
            "minx_miny_topz",
            "minx_maxy_topz",
            "maxx_miny_topz",
            "maxx_maxy_topz",
            "center_topz",
        ]

    scores = {}
    origins = {}
    for mode in candidates:
        origin = _stl_origin_top_surface(stl_verts, mode)
        verts_aligned = stl_verts - origin
        info = _score_alignment_top_surface(
            cnc_xyz,
            verts_aligned,
            eps_xy=eps_xy,
            eps_z=eps_z,
        )
        scores[mode] = info
        origins[mode] = origin

    best_mode = max(scores.keys(), key=lambda k: scores[k]["score"])
    best_origin = origins[best_mode]
    best_verts = stl_verts - best_origin

    if verbose:
        print("\n[AutoAlignTop] candidate scores (higher is better):")
        for k in candidates:
            v = scores[k]
            label = CANDIDATE_LABELS.get(k, k)
            print(
                f"  - {k:14s} [{label}] score={v['score']:.4f} in_ratio={v['in_ratio']:.3f} "
                f"penalty={v['penalty']:.4f} quad={v['quad']:.3f}"
            )
        best_label = CANDIDATE_LABELS.get(best_mode, best_mode)
        print(f"[AutoAlignTop] selected: {best_mode} [{best_label}] origin={best_origin.tolist()}\n")

    return best_verts, best_mode, scores, best_origin


def color_by_ucl(series: pd.Series, ucl3: pd.Series, ucl5: pd.Series, allow_negative: bool) -> list:
    """
    Map values to discrete colors based on UCL thresholds.
    - base: deep blue
    - exceed 3?: orange
    - exceed 5?: red
    If allow_negative=True, use |value| against |ucl|.
    """
    base_color = "#b5b5fb"
    orange = "#ffe100"
    red = "#ff0000"
    colors = []
    for val, u3, u5 in zip(series, ucl3, ucl5):
        if allow_negative:
            val_cmp = abs(val)
            u3_cmp = abs(u3)
            u5_cmp = abs(u5)
        else:
            val_cmp = val
            u3_cmp = u3
            u5_cmp = u5
        if pd.isna(val_cmp) or pd.isna(u3_cmp) or pd.isna(u5_cmp):
            colors.append(base_color)
            continue
        if val_cmp > u5_cmp:
            colors.append(red)
        elif val_cmp > u3_cmp:
            colors.append(orange)
        else:
            colors.append(base_color)
    return colors


def build_figure(path_df: pd.DataFrame, verts: np.ndarray, faces: np.ndarray, line_colors, marker_colors, title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name="STL",
            color="lightgray",
            opacity=0.35,
            showscale=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=path_df["Offset_X"],
            y=path_df["Offset_Y"],
            z=path_df["Offset_Z"],
            mode="lines+markers",
            line=line_colors,
            marker=marker_colors,
            name="CNC Path",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Offset X",
            yaxis_title="Offset Y",
            zaxis_title="Offset Z",
            aspectmode="data",
        ),
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def main(
    data_csv: Path,
    ref_csv: Path,
    stl_path: Path,
    auto_align_top: bool,
    eps_xy: float,
    eps_z: float,
    align_candidates: str,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path_df = load_and_merge(data_csv, ref_csv)
    verts, faces = stl_to_mesh3d(stl_path)
    # ç¸®å? STL å°ºåº¦ï¼ˆä?å¦?0.1 è¡¨ç¤ºç¸®å? 10 ?ï?
    STL_SCALE = 0.1
    verts = verts * STL_SCALE

    # STL alignment:
    # - Default behavior keeps legacy visualization (recenter to XY center, top Z at 0).
    # - With --auto-align-top, we instead align STL origin under top-surface-only assumption,
    #   choosing the best corner/center that fits CNC Offset points.
    if auto_align_top:
        candidates = [c.strip() for c in align_candidates.split(",") if c.strip()]
        cnc_xyz = path_df[["Offset_X", "Offset_Y", "Offset_Z"]].to_numpy(dtype=float)
        verts, best_mode, debug_scores, best_origin = auto_align_stl_top_surface(
            verts,
            cnc_xyz,
            candidates=candidates,
            eps_xy=eps_xy,
            eps_z=eps_z,
            verbose=True,
        )
    else:
        verts = recenter_stl_to_xy_center_top_z(verts)
    # Common styles
    line_template = {"width": 6}
    marker_template = {"size": 2}

    # 1) Torque value gradient
    fig1 = build_figure(
        path_df,
        verts,
        faces,
        line_colors=dict(color=path_df["Torque"], colorscale=COLOR_SCALE, width=line_template["width"], colorbar=dict(title="Torque")),
        marker_colors=dict(size=marker_template["size"], color=path_df["Torque"], colorscale=COLOR_SCALE),
        title="CNC Path on STL - Torque value gradient",
    )
    out1 = OUTPUT_DIR / "path_on_stl_torque_value.html"
    fig1.write_html(str(out1), include_plotlyjs="cdn", auto_open=False)

    # 2) Torque UCL discrete colors
    torque_colors = color_by_ucl(path_df["Torque"], path_df["Torque_UCL_3"], path_df["Torque_UCL_5"], allow_negative=False)
    fig2 = build_figure(
        path_df,
        verts,
        faces,
        line_colors=dict(color=torque_colors, width=line_template["width"]),
        marker_colors=dict(size=marker_template["size"], color=torque_colors),
        title="CNC Path on STL - Torque UCL",
    )
    out2 = OUTPUT_DIR / "path_on_stl_torque_ucl.html"
    fig2.write_html(str(out2), include_plotlyjs="cdn", auto_open=False)

    # 3) BendingX value gradient
    bend_abs = path_df["BendingX"].abs()
    fig3 = build_figure(
        path_df,
        verts,
        faces,
        line_colors=dict(
            color=bend_abs,
            colorscale=COLOR_SCALE,
            cmin=0,
            cmax=bend_abs.max() if not bend_abs.empty else 1,
            width=line_template["width"],
            colorbar=dict(title="|BendingX|"),
        ),
        marker_colors=dict(
            size=marker_template["size"],
            color=bend_abs,
            colorscale=COLOR_SCALE,
            cmin=0,
            cmax=bend_abs.max() if not bend_abs.empty else 1,
        ),
        title="CNC Path on STL - BendingX value gradient (|x|)",
    )
    out3 = OUTPUT_DIR / "path_on_stl_bending_value.html"
    fig3.write_html(str(out3), include_plotlyjs="cdn", auto_open=False)

    # 4) BendingX UCL discrete colors (two-sided)
    bend_colors = color_by_ucl(path_df["BendingX"], path_df["BendingX_UCL_3"], path_df["BendingX_UCL_5"], allow_negative=True)
    fig4 = build_figure(
        path_df,
        verts,
        faces,
        line_colors=dict(color=bend_colors, width=line_template["width"]),
        marker_colors=dict(size=marker_template["size"], color=bend_colors),
        title="CNC Path on STL - BendingX UCL (Â±)",
    )
    out4 = OUTPUT_DIR / "path_on_stl_bending_ucl.html"
    fig4.write_html(str(out4), include_plotlyjs="cdn", auto_open=False)

    print("Saved:")
    for p in [out1, out2, out3, out4]:
        print(f" - {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render CNC path on STL with reference UCL.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CSV, help="Aligned data CSV without UCL.")
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF_CSV, help="Reference CSV with UCL.")
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL, help="STL model path.")
    # Auto alignment: align STL to CNC path under the assumption that Z=0 is at the STL top surface (maxZ).
    parser.add_argument(
        "--auto-align-top",
        action="store_true",
        help="Auto-align STL origin (top-surface only) to best fit CNC Offset points (tries 4 corners + center).",
    )
    parser.add_argument(
        "--eps-xy",
        type=float,
        default=0.05,
        help="Tolerance (in STL/CNC units) for XY when scoring auto alignment (default: 0.05).",
    )
    parser.add_argument(
        "--eps-z",
        type=float,
        default=0.05,
        help="Tolerance (in STL/CNC units) for Z when scoring auto alignment (default: 0.05).",
    )
    parser.add_argument(
        "--align-candidates",
        type=str,
        default="minx_miny_topz,minx_maxy_topz,maxx_miny_topz,maxx_maxy_topz,center_topz",
        help="Comma-separated candidate modes for --auto-align-top.",
    )

    args = parser.parse_args()
    chosen_auto_align_top = HARDCODE_AUTO_ALIGN_TOP if HARDCODE_AUTO_ALIGN_TOP is not None else args.auto_align_top
    main(
        args.data,
        args.ref,
        args.stl,
        chosen_auto_align_top,
        args.eps_xy,
        args.eps_z,
        args.align_candidates,
    )
