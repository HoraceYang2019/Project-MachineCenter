import json
import threading
from pathlib import Path
import csv

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, dcc, html
import numpy as np


ROOT = Path(__file__).parent
DATA_FILE = ROOT / "CNC_opcua.json"
CSV_FILE = ROOT / "machining_6stage.csv"
MQTT_TOPIC = "cnc/snapshot"
MQTT_BROKER = "localhost" 
MQTT_PORT = 1883

RaWARN_LEVEL = 0.9
RaDANGER_LEVEL = 1.35

STATE_LOCK = threading.Lock()
LATEST_DATA = {}
LATEST_HISTORY = []
STL_CACHE = None



def list_stl_files():
    candidates = []
    for pattern in ("*.stl", "*.STL"):
        for stl_path in sorted(ROOT.glob(pattern)):
            candidates.append(stl_path)
    data_dir = ROOT / "stl_cell_dashboard" / "data"
    for pattern in ("*.stl", "*.STL"):
        if data_dir.exists():
            for stl_path in sorted(data_dir.glob(pattern)):
                candidates.append(stl_path)
    unique_paths = []
    seen = set()
    for stl_path in candidates:
        key = str(stl_path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(stl_path)
    return [{"label": stl_path.name, "value": str(stl_path)} for stl_path in unique_paths]


STL_OPTIONS = list_stl_files()


def point_color_for_ra(value):
    global RaWARN_LEVEL, RaDANGER_LEVEL
    RaWARN_LEVEL = RaWARN_LEVEL
    RaDANGER_LEVEL = RaDANGER_LEVEL
    if value is None:
        return "#6b7280"
    if value >= RaDANGER_LEVEL:   
        return "#d0021b"
    if value >= RaWARN_LEVEL:
        return "#f5a623"
    return "#2df945"


def build_stl_figure(stl_path=None, measurement_snapshot=None):
    default_path = ROOT / "!Back Plate.stl"
    target_path = Path(stl_path) if stl_path else default_path
    if not target_path.exists():
        target_path = default_path

    try:
        import trimesh

        mesh = trimesh.load_mesh(str(target_path))
        if getattr(mesh, "is_empty", False):
            raise ValueError("empty mesh")

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        vertices = mesh.vertices
        faces = mesh.faces
        x, y, z = vertices.T
        i, j, k = faces.T
        figure = go.Figure(
            data=[
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    opacity=0.55,
                    color="#6a8caf",
                )
            ]
        )
        if measurement_snapshot:
            point_values = extract_ra_points(measurement_snapshot)
            xmin = float(np.min(vertices[:, 0]))
            xmax = float(np.max(vertices[:, 0]))
            ymin = float(np.min(vertices[:, 1]))
            ymax = float(np.max(vertices[:, 1]))
            zmax = float(np.max(vertices[:, 2]))
            width = max(xmax - xmin, 1e-6)
            depth = max(ymax - ymin, 1e-6)
            x_positions = np.linspace(xmin + width * 0.15, xmax - width * 0.15, len(point_values))
            y_positions = np.linspace(ymin + depth * 0.25, ymin + depth * 0.75, len(point_values))
            z_positions = np.full(len(point_values), zmax + max((zmax - float(np.min(vertices[:, 2]))) * 0.02, 0.01))
            marker_colors = [point_color_for_ra(value) for value in point_values]
            # 標籤改為 P1~P6，並在下一行顯示 RA 值，保持簡潔同時提供必要資訊
            labels = [f"P{index + 1}<br>{value:.2f}" for index, value in enumerate(point_values)]
            figure.add_trace(
                go.Scatter3d(
                    x=x_positions,
                    y=y_positions,
                    z=z_positions,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    marker=dict(size=7, color=marker_colors, opacity=0.95),
                    name="量測點",
                    hovertemplate="check point %{text}<br>X %{x:.3f}<br>Y %{y:.3f}<br>Z %{z:.3f}<extra></extra>",
                )
            )
        figure.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=800)
        figure.update_scenes(aspectmode="data")
        return figure
    except Exception:
        fallback = go.Figure()
        fallback.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=800,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        fallback.add_annotation(
            text=f"STL 檢視需安裝 trimesh，或檔案不存在：{target_path.name}",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="#666"),
        )
        return fallback


def load_csv_snapshots(csv_path=CSV_FILE):
    if not csv_path.exists():
        return []
    snapshots = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            snapshots.append(dict(row))
    return snapshots


def build_csv_window(snapshots, current_index):
    if not snapshots:
        return []
    n = len(snapshots)
    current_index = max(0, min(current_index, n - 1))
    start = max(0, current_index - 3)
    end = min(n - 1, current_index + 2)
    return [snapshots[i] for i in range(start, end + 1)]

# RA盒狀圖
def compute_window_indices(snapshots, current_index):
    if not snapshots:
        return []
    n = len(snapshots)
    current_index = max(0, min(current_index, n - 1))
    start = max(0, current_index - 3)
    end = min(n - 1, current_index + 2)
    return list(range(start, end + 1))


def compute_window_labels(snapshots, current_index):
    # 取得當前時間窗口內的所有絕對索引清單（例如 [0, 1, 2, 3, 4, 5]）
    inds = compute_window_indices(snapshots, current_index)
    
    # 如果有 CSV 數據，就用索引去查出真正的 workpiece_id；查不到才用數字代替
    if snapshots:
        return [str(snapshots[i].get("workpiece_id", i)) for i in inds]
    return [str(i) for i in inds]


def load_data():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    return {}


def current_data():
    with STATE_LOCK:
        return dict(LATEST_DATA)


def current_history():
    with STATE_LOCK:
        return list(LATEST_HISTORY)


def update_state(new_data):
    if isinstance(new_data, dict) and isinstance(new_data.get("snapshots"), list):
        history = [item for item in new_data.get("snapshots", []) if isinstance(item, dict)]
    elif isinstance(new_data, list):
        history = [item for item in new_data if isinstance(item, dict)]
    elif isinstance(new_data, dict):
        history = [new_data]
    else:
        history = []

    with STATE_LOCK:
        LATEST_HISTORY.clear()
        LATEST_HISTORY.extend(history)
        LATEST_DATA.clear()
        if history:
            LATEST_DATA.update(history[-1])


def ensure_six_values(values, default_value):
    normalized = list(values[-6:]) if values else []
    if not normalized:
        normalized = [default_value] * 6
    while len(normalized) < 6:
        normalized.insert(0, normalized[0])
    return normalized[-6:]


def history_series(history, key, default_value):
    values = [to_float(item.get(key, default_value), default_value) for item in history if isinstance(item, dict)]
    return ensure_six_values(values, default_value)


def extract_ra_points(snapshot):
    points = []
    for i in range(1, 7):
        key = f"RA_P{i}"
        if key not in snapshot:
            continue
        value = snapshot.get(key)
        if value in (None, ""):
            continue
        points.append(to_float(value, 0.0))
    return points


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def state_badge(text, tone):
    return html.Div(text, className=f"status-badge status-{tone}")


def metric_card(title, value, unit, tone, subtitle="", card_id=None):
    value_id = f"{card_id}-value" if card_id else None
    subtitle_id = f"{card_id}-subtitle" if card_id else None
    badge_id = f"{card_id}-badge" if card_id else None
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="metric-title"),
                html.Div(f"{value}{unit}", id=value_id, className="metric-value"),
                html.Div(subtitle, id=subtitle_id, className="metric-subtitle"),
                html.Div(tone, id=badge_id, className=f"status-badge status-{tone}"),
            ]
        ),
        className="metric-card",
    )

# 
def build_trend_figure(title, base_value, warn_level, danger_level, batch_offset):
    labels = [f"I{offset:+d}" if offset else "I" for offset in range(-3, 3)]
    x_values = list(range(len(labels)))
    y_values = []
    for index in x_values:
        offset = index - 3
        batch_shift = batch_offset * 0.03
        value = max(0.0, base_value + offset * base_value * 0.02 + batch_shift + (index % 2) * 0.01)
        y_values.append(value)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values[:4],
            y=y_values[:4],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=3),
            name="過去 / 現在",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values[3:],
            y=y_values[3:],
            mode="lines+markers",
            line=dict(color="#d62728", width=3, dash="dash"),
            name="預測",
        )
    )
    figure.add_hline(y=warn_level, line_dash="dot", line_color="#f5a623")
    figure.add_hline(y=danger_level, line_dash="dot", line_color="#d0021b")
    figure.update_layout(
        title=title,
        margin=dict(l=16, r=16, t=42, b=16),
        height=230,
        showlegend=False,
        xaxis=dict(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
        yaxis_title="數值",
    )
    return figure


def build_history_trend_figure(title, values, warn_level, danger_level, labels=None, now_pos=None, x_values=None):
    if labels is None:
        labels = [str(i) for i in range(len(values))]
    if x_values is None:
        x_values = list(range(len(values)))
    figure = go.Figure()
    split_at = (now_pos + 1) if (now_pos is not None and 0 <= now_pos < len(x_values)) else (4 if len(x_values) >= 4 else len(x_values))
    # 判斷是否有 NOW 點
    if now_pos is not None and 0 <= now_pos < len(x_values):
        
        # 1. 已加工部分 (包含 NOW 這個點，這樣線才會連到 NOW)
        if now_pos >= 0:
            figure.add_trace(
                go.Scatter(
                    x=x_values[:now_pos + 1], # 多取一格到 now_pos
                    y=values[:now_pos + 1],
                    mode="lines+markers+text", # 加入 text 模式標示點
                    text=[f"{v:.2f}" for v in values[:now_pos + 1]], # 標示數值
                    textposition="top center",
                    line=dict(color="#1f77b4", width=3),
                    marker=dict(color="#1f77b4", size=8),
                    name="before",
                )
            )

        # 2. 推測未來部分 (從 NOW 這個點開始，線才會從 NOW 出發)
        if now_pos < len(x_values) - 1:
            figure.add_trace(
                go.Scatter(
                    x=x_values[now_pos:], # 從 now_pos 開始
                    y=values[now_pos:],
                    mode="lines+markers",
                    line=dict(color="#f39c12", width=3, dash="dash"), # 預測用虛線
                    marker=dict(color="#f39c12", size=8),
                    name="after",
                )
            )
            
        # 3. 特別標出 NOW 這個點 (大圓點)
        figure.add_trace(
            go.Scatter(
                x=[x_values[now_pos]],
                y=[values[now_pos]],
                mode="markers",
                marker=dict(color="#2ecc71", size=14, symbol="circle", line=dict(width=2, color="white")),
                name="NOW",
                hoverinfo="skip" # 避免重複顯示 hover
            )
        )
    else:
        # 如果沒有 NOW 點，就畫一條完整的實線
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=values,
                mode="lines+markers+text",
                text=[f"{v:.2f}" for v in values],
                textposition="top center",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
            )
        )
    if now_pos is not None and 0 <= now_pos < len(x_values) and now_pos < len(labels):
        selected_x = x_values[now_pos]
        figure.add_vline(x=selected_x, line_color="#2ecc71", line_width=3, line_dash="solid")
        figure.add_annotation(x=selected_x, y=1.02, xref="x", yref="paper", text="NOW", showarrow=False, font=dict(color="#2ecc71", size=12))
        if now_pos > 0:
            figure.add_annotation(x=x_values[0], y=1.02, xref="x", yref="paper", text="before", showarrow=False, font=dict(color="#1f77b4", size=12))
        if now_pos + 1 < len(x_values):
            figure.add_annotation(x=x_values[-1], y=1.02, xref="x", yref="paper", text="after", showarrow=False, font=dict(color="#f39c12", size=12))
    # 取得workpiece_id對應的標籤
    if labels is None:
        labels = [str(i) for i in range(len(x_values))]

    figure.add_hline(y=warn_level, line_dash="dot", line_color="#f5a623")
    figure.add_hline(y=danger_level, line_dash="dot", line_color="#d0021b")
    figure.update_layout(
        title=title,
        margin=dict(l=16, r=16, t=42, b=16),
        height=230,
        autosize=False,
        showlegend=False,
        xaxis=dict(tickmode="array", tickvals=x_values, ticktext=labels),
        yaxis_title="value",
    )
    return figure


def build_heatmap_figure(batch_offset , labels=None):
    labels = [str(i) for i in range(6)] if labels is None else labels
    categories = ["broken", "heavy", "normal", "light"]
    matrix = []
    for row_index, _category in enumerate(categories):
        row = []
        for col_index, _label in enumerate(labels):
            base = [0.12, 0.18, 0.52, 0.18][row_index]
            value = max(0.01, base + (col_index - 2) * 0.03 + batch_offset * 0.005 * (row_index + 1))
            row.append(value)
        row_total = sum(row)
        matrix.append([item / row_total for item in row])

    figure = px.imshow(
        matrix,
        x=labels,
        y=categories,
        color_continuous_scale=[
            [0.0, "#2d7ff9"],
            [0.33, "#2ecc71"],
            [0.66, "#f1c40f"],
            [1.0, "#e74c3c"],
        ],
        aspect="auto",
        labels=dict(x="工件視窗", y="Status", color="Probability"),
    )
    figure.update_layout(title="Risk of Failure", margin=dict(l=16, r=16, t=42, b=16), height=230, autosize=False)
    return figure


def build_history_heatmap_figure(history, labels=None, now_pos=None):
    global RaWARN_LEVEL, RaDANGER_LEVEL
    
    if labels is None:
        labels = [str(i) for i in range(len(history))] if history else [str(i) for i in range(6)]
        
    # 定義新的分類
    categories = ["good", "normal", "bad"]
    
    if not history:
        return go.Figure()

    # 定義對應顏色，與 point_color_for_ra 保持視覺一致性
    palette = {
        "bad": "#d0021b",     # 紅色 (>= 1.35)
        "normal": "#f5a623",  # 橘色 (>= 0.9)
        "good": "#3bc74c",    # 綠色 (< 0.9)
    }
    
    # 收集每個狀態的數量變化
    data_series = {cat: [] for cat in categories}
    
    for snapshot in history:
        counts = {"good": 0, "normal": 0, "bad": 0}
        
        if isinstance(snapshot, dict):
            pts = extract_ra_points(snapshot)
            if pts:
                # 統計該批次 6 個點的 RA 值落在哪個區間
                for val in pts:
                    if val >= RaDANGER_LEVEL:
                        counts["bad"] += 1
                    elif val >= RaWARN_LEVEL:
                        counts["normal"] += 1
                    else:
                        counts["good"] += 1
            else:
                # 若該批次無量測資料，您可以決定是否要給預設值。此處以 0 計算。
                pass

        for cat in categories:
            data_series[cat].append(counts[cat])

    figure = go.Figure()

    # 繪圖順序：good 放底部，normal 疊加上層，bad 放最頂部以凸顯風險
    plot_order = ["good", "normal", "bad"]
    
    for cat in plot_order:
        figure.add_trace(
            go.Bar(
                x=list(range(len(labels))),
                y=data_series[cat],
                name=cat,
                marker_color=palette[cat],
                hovertemplate=f"{cat}<br>數量: %{{y}} 點<extra></extra>"
            )
        )

    # 標示 NOW 與輔助線
    if now_pos is not None and 0 <= now_pos < len(labels):
        figure.add_vline(x=now_pos, line_color="#34495e", line_width=2, line_dash="dash")
        figure.add_annotation(x=now_pos, y=1.08, xref="x", yref="paper", text="NOW", showarrow=False, font=dict(color="#34495e", size=12))
        if now_pos > 0:
            figure.add_annotation(x=0, y=1.08, xref="x", yref="paper", text="before", showarrow=False, font=dict(color="#1f77b4", size=12))
        if now_pos + 1 < len(labels):
            figure.add_annotation(x=len(labels) - 1, y=1.08, xref="x", yref="paper", text="after", showarrow=False, font=dict(color="#f39c12", size=12))

    figure.update_layout(
        title="Risk of Failure (RA Statistics)",
        barmode="stack",  
        margin=dict(l=16, r=16, t=42, b=16),
        height=260,
        autosize=False,
        # 將 Y 軸改為顯示實際數量 (整數)，並移除原本的百分比限制
        yaxis=dict(title="Point Count", tickformat="d"), 
        xaxis=dict(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5) 
    )
    
    return figure



def build_box_figure(base_value, batch_offset):
    series = []
    for index in range(6):
        sample = [max(0.0, base_value + batch_offset * 0.01 + (index - 2) * 0.08 + (n - 6) * 0.04 + (n % 3) * 0.01) for n in range(12)]
        series.append(sample)
    figure = go.Figure()
    for index, sample in enumerate(series):
        label = f"I{index - 3:+d}" if index - 3 else "I"
        figure.add_trace(go.Box(y=sample, name=label, marker_color="#6a8caf"))
    figure.update_layout(title="Quality- Ra Statistics", margin=dict(l=16, r=16, t=42, b=16), height=230, autosize=False, showlegend=False)
    return figure


def build_history_box_figure(history, key, default_value, labels=None, now_pos=None):
    global RaWARN_LEVEL, RaDANGER_LEVEL
    # history: 列表，為 build_csv_window 返回的快照序列（不環回）
    figure = go.Figure()
    # 逐一檢查 history 中的每一個位置；有多少個 RA_P* 就畫多少個，不補值、不推算
    for idx, snap in enumerate(history):
        if not isinstance(snap, dict):
            continue
        pts = extract_ra_points(snap)
        if not pts:
            continue
        label = labels[idx] if (labels and idx < len(labels)) else f"Stage {idx}"
        # NOW 位置用特殊顏色
        if now_pos is not None and idx < now_pos:
            marker_color = "#1f77b4"
        elif now_pos is not None and idx == now_pos:
            marker_color = "#2ecc71"
        elif now_pos is not None and idx > now_pos:
            marker_color = "#f39c12"
        else:
            marker_color = "#6a8caf"
        # 使用數字 x 座標對齊位置，避免類別軸與 vline 不對齊
        figure.add_trace(go.Box(x=[idx] * len(pts), y=pts, name=label, marker_color=marker_color))

    if now_pos is not None and labels and 0 <= now_pos < len(labels):
        # vline/annotations 使用數字座標（位置索引），x 軸顯示文字標籤
        figure.add_vline(x=now_pos, line_color="#2ecc71", line_width=3, line_dash="solid")
        figure.add_annotation(x=now_pos, y=1.02, xref="x", yref="paper", text="NOW", showarrow=False, font=dict(color="#2ecc71", size=12))
        if now_pos > 0:
            figure.add_annotation(x=0, y=1.02, xref="x", yref="paper", text="before", showarrow=False, font=dict(color="#1f77b4", size=12))
        if now_pos + 1 < len(labels):
            figure.add_annotation(x=len(labels) - 1, y=1.02, xref="x", yref="paper", text="after", showarrow=False, font=dict(color="#f39c12", size=12))
    
    figure.add_hline(y=RaDANGER_LEVEL, line_dash="dash", line_color="#d0021b")
    figure.add_hline(y=RaWARN_LEVEL, line_dash="dot", line_color="#f5a623")
    figure.update_layout(
        title="Ra Statistics",
        margin=dict(l=16, r=16, t=42, b=16),
        height=230,
        autosize=False,
        showlegend=False,
        yaxis_title="um",
        xaxis_title="Workpiece No.",
    )
    # 設定 x 軸顯示的文字標籤
    if labels:
        figure.update_xaxes(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels)
    return figure


def build_ra_histogram_figure(base_value, batch_offset):
    points = [max(0.0, base_value + batch_offset * 0.01 + (index - 10) * 0.02 + (index % 4) * 0.005) for index in range(30)]
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=points, nbinsx=10, marker_color="#1f77b4"))
    figure.add_vline(x=base_value + 0.8, line_dash="dash", line_color="#f5a623")
    figure.add_vline(x=base_value + 1.5, line_dash="dash", line_color="#d0021b")
    figure.update_layout(title="RA Histogram", margin=dict(l=16, r=16, t=42, b=16), height=230, autosize=False, showlegend=False, xaxis_title="推論品質", yaxis_title="數量")
    return figure


def build_history_ra_histogram_figure(snapshot, base_value):
    # 顯示當前（NOW）快照中實際存在的 RA_P 點位
    if isinstance(snapshot, dict):
        points = extract_ra_points(snapshot)
    else:
        points = []
    labels = [f"P{i + 1}" for i in range(len(points))]
    figure = go.Figure()
    if points:
        if points:
        # 將 go.Bar 改為 go.Scatter
            figure.add_trace(go.Scatter(
            x=labels, 
            y=points, 
            mode="lines+markers",  # 顯示線段和點
            line=dict(
                color="#1f77b4", 
                width=3, 
                shape='spline'     # 讓線段變成平滑曲線
            ),
            marker=dict(
                size=8, 
                color="#1f77b4",
                symbol="circle"
            ),
            name="RA 數值"
        ))
    else:
        figure.add_annotation(
            text="目前沒有可用的 RA_P1~RA_P6 資料",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="#666"),
        )
    global RaWARN_LEVEL, RaDANGER_LEVEL
    figure.add_hline(y=RaWARN_LEVEL, line_dash="dot", line_color="#f5a623")
    figure.add_hline(y=RaDANGER_LEVEL, line_dash="dash", line_color="#d0021b")
    figure.update_layout(
        title="Quality- Ra Values", 
        margin=dict(l=16, r=16, t=42, b=16),
        height=230,
        autosize=False,
        showlegend=False,
        xaxis_title="Sampling No.",
        yaxis_title="um",
    )
    return figure


def build_status_gauge_figure(batch_offset):
    probability = max(0.0, min(1.0, 0.45 + batch_offset * 0.02))
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Status Probability / Needle"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 25], "color": "#fde2e2"},
                    {"range": [25, 50], "color": "#fff3cd"},
                    {"range": [50, 75], "color": "#e7f7ec"},
                    {"range": [75, 100], "color": "#dbeafe"},
                ],
            },
        )
    )
    figure.update_layout(margin=dict(l=16, r=16, t=42, b=16), height=230, autosize=False)
    return figure


def build_history_status_gauge_figure(history):
    latest = history[-1] if history else {}
    status = str(latest.get("Status", "OK")).upper()
    probability = 82 if status == "WARN" else 46
    steps = [
        {"range": [25, 50], "color": "#fff3cd"},
        {"range": [50, 75], "color": "#e7f7ec"},
        {"range": [75, 100], "color": "#dbeafe"},
    ]
    if status == "WARN":
        steps.insert(0, {"range": [0, 25], "color": "#fde2e2"})
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%"},
            title={"text": "Status Probability / Needle"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": steps,
            },
        )
    )
    figure.update_layout(margin=dict(l=8, r=8, t=28, b=8), height=170, autosize=False)
    return figure


def connect_mqtt():
    try:
        import paho.mqtt.client as mqtt
    except Exception:
        return None

    client = mqtt.Client(client_id="cnc_dash_viewer")

    def on_connect(client, userdata, flags, rc):
        try:
            client.subscribe(MQTT_TOPIC)
        except Exception:
            pass

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            update_state(payload)
        except Exception:
            return

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        return client
    except Exception:
        return None


if not LATEST_DATA:
    update_state(load_data())

CSV_SNAPSHOTS = load_csv_snapshots()
if CSV_SNAPSHOTS:
    update_state({"snapshots": CSV_SNAPSHOTS})


# 建立滑桿標記字典
slider_marks = {}

if CSV_SNAPSHOTS:
    for i, snap in enumerate(CSV_SNAPSHOTS):
        w_id = str(snap.get("workpiece_id", i))
        
        # 1. 擷取該工件的 RA 數值（相容原本的替代欄位 Offset_Z）
        ra_val = to_float(snap.get("RA", snap.get("Offset_Z", 0.0)), 0.0)
        
        # 2. 取得對應的顏色（可使用您現有的 point_color_for_ra 函數）
        # 如果想採用先前調整過的專業工業色票，可以參考下方的對應：
        # 正常 (<0.8) -> #43A047 | 警戒 (>=0.8) -> #F57C00 | 斷刀/危險 (>=1.5) -> #C62828
        mark_color = point_color_for_ra(ra_val)
        
        # 3. 封裝成 Dash 支援的樣式字典
        slider_marks[i] = {
            "label": w_id,
            "style": {
                "color": mark_color, 
                "fontWeight": "700",       # 加粗字體讓顏色更明顯
                "fontSize": "12px"
            }
        }
else:
    slider_marks = {0: "None"}

csv_count = len(CSV_SNAPSHOTS)
csv_max = max(0, csv_count - 1)

MQTT_CLIENT = connect_mqtt()

FA_URL = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder=str(ROOT / "assets"))

workpiece_ids = [str(snap.get("workpiece_id", "")) for snap in CSV_SNAPSHOTS if snap.get("workpiece_id")]
csv_count = len(workpiece_ids)
csv_max = max(0, csv_count - 1)
# 建立滑桿位置與實際工件識別碼的對應字典，作為 Slider 的 marks
slider_marks = {i: w_id for i, w_id in enumerate(workpiece_ids)} if csv_count > 0 else {0: "None"}

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H2("- Workspace"), md=10),
                dbc.Col(dbc.Button("返回", id="return-btn", color="light", className="w-100"), md=2),
            ],
            align="center",
            className="my-2",
        ),
        dcc.Interval(id="mqtt-refresh", interval=1000, n_intervals=0),
        dbc.Row(
            [
                dbc.Col(metric_card("Tool Wear", "0.25", "mm", "ok", "警戒值", card_id="wear"), md=2),
                dbc.Col(metric_card("Torque", "0.8", "Nm", "warn", "警戒值", card_id="toque"), md=2),
                dbc.Col(metric_card("Bending", "80 / 150", "", "ok", "上下界", card_id="bending"), md=2),
                dbc.Col([dcc.Graph(id="status-gauge", config={"displayModeBar": False, "responsive": False}, style={"height": "170px"})], md=2, className="status-gauge-card"),
                dbc.Col(metric_card("Ra", "1.5", "um", "warn", "警戒值", card_id="ra"), md=4),
            ],
            className="g-2 mb-3",
        style={"display": "none"},
        ),
        html.Div(
            [
                html.Div(
                [
                    dcc.Graph(id="wear-figure", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                    dcc.Graph(id="toque-figure", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                    dcc.Graph(id="bending-figure", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                    
                    # --- 新增：Tool States by TID ---
                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Tool States by TID", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                            html.Div(id="tool-states-content", style={"fontSize": "14px", "minHeight": "60px"})
                        ]),
                        className="mt-2", style={"boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
                    ),
                    
                    # --- 新增：Next Responses for Tool Issues ---
                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Next Responses for Tool Issues", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                            html.Div(id="tool-response-content", style={"fontSize": "14px", "minHeight": "60px"})
                        ]),
                        className="mt-2", style={"boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
                    ),
                ],
                className="left-panel",
            ),
                html.Div(
                    [
                        # --- 新增/替換：States Summary 頂部狀態列 ---
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "padding": "10px 16px",
                                "backgroundColor": "#f8f9fa",
                                "border": "1px solid #dee2e6",
                                "borderRadius": "4px",
                                "marginBottom": "12px"
                            },
                            children=[
                                # 左側：多把刀具狀態燈號
                                html.Div(id="summary-tools-state", style={"display": "flex", "gap": "8px", "alignItems": "center", "flex": "1"}),
                                
                                # 中間： 工件名稱
                                html.Div(
                                    children=[
                                        html.Span(style={"fontWeight": "bold", "fontSize": "16px", "color": "#333"}),
                                        html.Span(id="summary-workpiece-name", style={"fontWeight": "bold", "fontSize": "16px", "color": "#1f77b4"})
                                    ],
                                    style={"textAlign": "center", "flex": "1"}
                                ),
                                
                                # 右側：佔位符，確保中間區塊能絕對置中
                                html.Div(style={"flex": "1"})
                            ]
                        ),
                        html.Div(
                            style={"display": "flex", "flexDirection": "column", "alignItems": "stretch"},
                            children=[
                                html.Div(
                                    style={"position": "relative", "minHeight": "220px", "width": "100%"},
                                    children=[
                                        dcc.Graph(
                                            id="stl-graph",
                                            figure=build_stl_figure(STL_OPTIONS[0]["value"] if STL_OPTIONS else None),
                                            config={"displayModeBar": False, "responsive": False},
                                            style={"height": "800px", "width": "100%"},
                                        ),
                                        html.Div(
                                            style={"position": "absolute", "top": "8px", "left": "8px", "width": "220px", "zIndex": 5,"disabled":"none"},
                                            children=[
                                                dcc.Dropdown(id="stl-dropdown", options=STL_OPTIONS, placeholder="Select STL", persistence=True, value=(STL_OPTIONS[0]["value"] if STL_OPTIONS else None) ),
                                                html.Div("Part View", style={"marginTop": "6px", "fontSize": "12px", "fontWeight": "600"}),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(style={"marginTop": "12px"},children=[
                                    dbc.Row(
                                    [   
                                        dbc.Col(
                                            [
                                                dcc.Slider(
                                                    id="workpiece-slider",  # 確保與您的 Callback Input 一致
                                                    min=0,
                                                    max=csv_max,
                                                    step=1,
                                                    value=0,
                                                    marks=slider_marks,     # 帶入含有顏色樣式的字典
                                                    disabled=(csv_count == 0),
                                                )
                                            ],
                                            md=10,
                                        ),
                                    ],
                                    className="mb-3 justify-content-center",
                                ),]),
                                # --- 新增/替換：NC Path List ---
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H6("NC Path List", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                                        html.Div(id="nc-path-content", style={"fontSize": "14px", "minHeight": "60px", "fontFamily": "monospace", "backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "4px"})
                                    ]),
                                    className="mt-2", style={"boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
                                ),
                            ],
                        ),
                    ],
                    className="center-panel",
                ),
                html.Div(
                [
                    dcc.Graph(id="status-heatmap", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                    dcc.Graph(id="ra-box", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                    dcc.Graph(id="ra-trend", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                    
                    # --- 新增：Quality States by WID ---
                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Quality States by WID", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                            html.Div(id="quality-states-content", style={"fontSize": "14px", "minHeight": "60px"})
                        ]),
                        className="mt-2", style={"boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
                    ),
                    
                    # --- 新增：Next Responses for Quality Issues ---
                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Next Responses for Quality Issues", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                            html.Div(id="quality-response-content", style={"fontSize": "14px", "minHeight": "60px"})
                        ]),
                        className="mt-2", style={"boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
                    ),
                ],
                className="right-panel",
            ),
            ],
            className="main-three-column",
        ),
        html.Div(id="raw-json-holder", children=html.Pre("{}", className="raw-json d-none")),
    ],
    fluid=True,
    className="page-root",
)




@app.callback(
    Output("summary-workpiece-name", "children"),
    Output("wear-figure", "figure"),
    Output("toque-figure", "figure"),
    Output("bending-figure", "figure"),
    Output("status-heatmap", "figure"),
    Output("ra-box", "figure"),
    Output("ra-trend", "figure"),
    Output("status-gauge", "figure"),
    Output("wear-value", "children"),
    Output("wear-subtitle", "children"),
    Output("wear-badge", "children"),
    Output("wear-badge", "className"),
    Output("toque-value", "children"),
    Output("toque-subtitle", "children"),
    Output("toque-badge", "children"),
    Output("toque-badge", "className"),
    Output("bending-value", "children"),
    Output("bending-subtitle", "children"),
    Output("bending-badge", "children"),
    Output("bending-badge", "className"),
    Output("ra-value", "children"),
    Output("ra-subtitle", "children"),
    Output("ra-badge", "children"),
    Output("ra-badge", "className"),
    Output("raw-json-holder", "children"),
    Output("tool-states-content", "children"),   # 新增輸出 1
    Output("tool-response-content", "children"), # 新增輸出 2
    Output("quality-states-content", "children"),   # 新增輸出 3
    Output("quality-response-content", "children"), # 新增輸出 4
    Output("nc-path-content", "children"),  # 新增輸出 5
    Output("summary-tools-state", "children"),  # 新增輸出 6
    Input("mqtt-refresh", "n_intervals"),
    Input("stl-dropdown", "value"),
    Input("workpiece-slider", "value"),
)
def update_figures(_n_intervals, stl_path, slider_value):    
    if CSV_SNAPSHOTS:
        # 2. 將滑桿目前的位置數字 (0, 1, 2...) 轉為安全的整數索引
        current_idx = to_int(slider_value, 0) % len(CSV_SNAPSHOTS)
        
        # 3. 確保能拿到對應位置的快照資料
        snapshot = CSV_SNAPSHOTS[current_idx]
        
        # 4. 沿用你原本建立的時間窗口計算邏輯
        history = build_csv_window(CSV_SNAPSHOTS, current_idx)
        window_indices = compute_window_indices(CSV_SNAPSHOTS, current_idx)
        window_labels = compute_window_labels(CSV_SNAPSHOTS, current_idx)
        start = window_indices[0] if window_indices else current_idx
        now_pos = current_idx - start if window_indices else None
    else:
        snapshot = current_data()
        history = current_history()
        window_indices = list(range(len(history)))
        window_labels = None
        now_pos = None

    # 5. 更新 batch_offset 的定義，使其與當前查到的 workpiece_id 連動（供模擬數據增量使用）
    wear_base = to_float(snapshot.get("Tool_Life", 150), 150)
    toque_base = to_float(snapshot.get("Toque", snapshot.get("Radius_Comp", 0.8)), 0.8)
    bending_base = to_float(snapshot.get("Bending", snapshot.get("Length_Comp", 80)), 80)
    ra_base = max(0.1, to_float(snapshot.get("RA", snapshot.get("Offset_Z", 1.5)), 1.5))
    tool_code = snapshot.get("Tool_Code", "-")
    status_text = str(snapshot.get("Status", "OK")).upper()

    wear_values = [to_float(item.get("Tool_Life", wear_base), wear_base) for item in history if isinstance(item, dict)]
    toque_values = [to_float(item.get("Toque", toque_base), toque_base) for item in history if isinstance(item, dict)]
    bending_values = [to_float(item.get("Bending", bending_base), bending_base) for item in history if isinstance(item, dict)]
    x_positions = list(range(len(history)))
    wear_figure = build_history_trend_figure("Tool Wear", wear_values, 0.25, 0.35, labels=window_labels, now_pos=now_pos, x_values=x_positions)
    toque_figure = build_history_trend_figure("Torque", toque_values, 0.23, 0.32, labels=window_labels, now_pos=now_pos, x_values=x_positions)
    bending_figure = build_history_trend_figure("Bending", bending_values, 7.7, 12.0, labels=window_labels, now_pos=now_pos, x_values=x_positions)
    heatmap_figure = build_history_heatmap_figure(history, labels=window_labels, now_pos=now_pos)
    box_figure = build_history_box_figure(history, "RA", ra_base, labels=window_labels, now_pos=now_pos)
    ra_trend_figure = build_history_ra_histogram_figure(snapshot, ra_base)
    gauge_figure = build_history_status_gauge_figure(history)
    wear_value = f"{wear_base:.2f}"
    toque_value = f"{toque_base:.2f}"
    bending_value = f"{bending_base:.0f} / 150"
    ra_value = f"{ra_base:.2f}"
    wear_subtitle = f"Tool_Code: {tool_code}"
    toque_subtitle = f"來源: {status_text}"
    bending_subtitle = "上下界"
    ra_subtitle = f"Status: {status_text}"
    wear_badge = "OK" if wear_base <= 0.25 else "WARN"
    toque_badge = "OK" if toque_base <= 0.8 else "WARN"
    bending_badge = "OK" if bending_base <= 80 else "WARN"
    ra_badge = "OK" if ra_base <= RaDANGER_LEVEL else "WARN"
    raw_json = html.Pre(json.dumps(snapshot, indent=2, ensure_ascii=False), className="raw-json d-none")
    # === 新增：Tool States by TID 邏輯 ===
    tool_state_elements = [html.Div(f"Current Tool: T{tool_code}")]
    if wear_base > 0.25:
        tool_state_elements.append(html.Div("State: Bending Torque Warning", style={"color": "#d0021b", "fontWeight": "bold"}))
    elif toque_base > 0.8:
        tool_state_elements.append(html.Div("State: Overload Warning", style={"color": "#f5a623", "fontWeight": "bold"}))
    else:
        tool_state_elements.append(html.Div("State: Normal", style={"color": "#43A047"}))

    # === 新增：Next Responses for Tool Issues 邏輯 ===
    response_elements = []
    if wear_base > 0.25:
        response_elements.append(html.Div("1. Bendingg、Torque受力過大，預測刀具壽命即將到達極限，請準備更換備品。"))
        response_elements.append(html.Div("2. 建議降低下一工件之進給率 (Feed Rate)。"))
    elif toque_base > 0.8:
        response_elements.append(html.Div("1. 主軸負載異常，請檢查切削液供應與排屑狀態。"))
    else:
        response_elements.append(html.Div("維持原 NC 程式參數進行加工。", style={"color": "#666"}))
    # === 新增：Quality States by WID 邏輯 ===
    current_wid = str(snapshot.get("workpiece_id", current_idx))
    pts = extract_ra_points(snapshot)
    # 取出該工件 6 個點中的最大 RA 值作為品質評級基準
    max_ra = max(pts) if pts else ra_base

    quality_state_elements = [html.Div(f"Current WID: {current_wid}")]
    if max_ra >= RaDANGER_LEVEL:
        quality_state_elements.append(html.Div(f"State: Out of Spec (Max Ra: {max_ra:.2f})", style={"color": "#d0021b", "fontWeight": "bold"}))
    elif max_ra >= RaWARN_LEVEL:
        quality_state_elements.append(html.Div(f"State: Warning (Max Ra: {max_ra:.2f})", style={"color": "#f5a623", "fontWeight": "bold"}))
    else:
        quality_state_elements.append(html.Div(f"State: Good (Max Ra: {max_ra:.2f})", style={"color": "#43A047"}))

    # === 新增：Next Responses for Quality Issues 邏輯 ===
    quality_response_elements = []
    if max_ra >= RaDANGER_LEVEL:
        quality_response_elements.append(html.Div("1. 立即暫停，檢查刀具表面磨損與崩刃狀況。"))
        quality_response_elements.append(html.Div("2. 確認工件夾持是否鬆動造成異常切削震動。"))
        quality_response_elements.append(html.Div("3. 重新執行刀長與刀徑尺寸補償 (Offset)。"))
    elif max_ra >= RaWARN_LEVEL:
        quality_response_elements.append(html.Div("1. 表面粗糙度逼近上限，建議檢查切削液。"))
        quality_response_elements.append(html.Div("2. 預防性微調 Z 軸補償值。"))
    else:
        quality_response_elements.append(html.Div("工件品質良好，無須變更對策。", style={"color": "#666"}))
    # === 新增：NC Path List 邏輯 ===
    nc_path_elements = []
    # === 新增：States Summary 刀具狀態邏輯 ===
    # 根據當前數據的磨耗與負載，決定主要刀具的背景顏色
    tool_status_color = "#3B963F"  # 預設正常(綠色)
    if wear_base > RaWARN_LEVEL or toque_base > 12.:
        tool_status_color = "#f5a623"  # 警戒(橘色)
    if wear_base >= RaDANGER_LEVEL or toque_base >= 0.3:
        tool_status_color = "#d0021b"  # 危險(紅色)

    # 建立多把刀具的標籤 (此處預設帶入當前刀號，並輔以模擬刀具 T10, T14)
    # 若您的 CSV 內有陣列記錄所有使用刀具，可替換為迴圈動態生成
    summary_tools_elements = [
        html.Div("Tools:", style={"fontSize": "14px", "fontWeight": "bold", "color": "#666"}),
        html.Div(f"T{tool_code}", style={"backgroundColor": tool_status_color, "color": "white", "padding": "2px 8px", "borderRadius": "12px", "fontSize": "12px", "fontWeight": "bold"}),
        html.Div("T10", style={"backgroundColor": "#43A047", "color": "white", "padding": "2px 8px", "borderRadius": "12px", "fontSize": "12px", "fontWeight": "bold"}),
        html.Div("T14", style={"backgroundColor": "#43A047", "color": "white", "padding": "2px 8px", "borderRadius": "12px", "fontSize": "12px", "fontWeight": "bold"})
    ]
    
# 1. 擷取工件名稱
    current_wid = str(snapshot.get("workpiece_id", current_idx))

    # 2. 擷取 STL 檔名 (透過 pathlib 取得純檔名)
    if stl_path:
        stl_name = Path(stl_path).name
    elif STL_OPTIONS: # 預設載入第一個檔案的名稱
        stl_name = STL_OPTIONS[0]["label"]
    else:
        stl_name = "未選擇 STL"

    # 3. 組合顯示字串
    center_display_text = f"{stl_name} - {current_wid}"
    # 判斷是否因 Torque 超標造成異常負載
    if toque_base > 0.3:
        nc_path_elements.append(html.Div("[警告] 異常負載發生於:", style={"color": "#d0021b", "fontWeight": "bold"}))
        # 這裡放入模擬或實際抓取的 NC G-code
        nc_path_elements.append(html.Div("N0140 G01 X15.5 Y20.0 Z-5.0 F150", style={"color": "#d0021b"}))
        nc_path_elements.append(html.Div(f"原因分析: Torque 數值 ({toque_base:.2f} Nm) 超出警戒線 (0.3 Nm)", style={"color": "#666", "fontSize": "12px", "marginTop": "4px"}))
        
    # 判斷是否因 Bending 超標造成異常受力
    elif bending_base > 12:
        nc_path_elements.append(html.Div("[警告] 異常Bending受力發生於:", style={"color": "#f5a623", "fontWeight": "bold"}))
        nc_path_elements.append(html.Div("N0160 G02 X20.0 Y25.0 I4.5 J0.0 F120", style={"color": "#f5a623"}))
        nc_path_elements.append(html.Div(f"原因分析: Bending 數值 ({bending_base:.0f}) 超過正常上下界限", style={"color": "#666", "fontSize": "12px", "marginTop": "4px"}))
        
    # 若皆正常
    else:
        nc_path_elements.append(html.Div("目前未偵測到異常負載的 NC 段落。", style={"color": "#43A047"}))
    return (
        center_display_text,        # 1. 輸出組合後的字串        
        wear_figure,
        toque_figure,
        bending_figure,
        heatmap_figure,
        box_figure,
        ra_trend_figure,
        gauge_figure,
        f"{wear_value} mm",
        wear_subtitle,
        wear_badge,
        f"status-badge status-{'ok' if wear_badge == 'OK' else 'warn'}",
        f"{toque_value} Nm",
        toque_subtitle,
        toque_badge,
        f"status-badge status-{'ok' if toque_badge == 'OK' else 'warn'}",
        bending_value,
        bending_subtitle,
        bending_badge,
        f"status-badge status-{'ok' if bending_badge == 'OK' else 'warn'}",
        f"{ra_value} um",
        ra_subtitle,
        ra_badge,
        f"status-badge status-{'ok' if ra_badge == 'OK' else 'warn'}",
        raw_json,
        tool_state_elements,  # 對應 tool-states-content
        response_elements,    # 對應 tool-response-content
        quality_state_elements,  # 對應 quality-states-content
        quality_response_elements, # 對應 quality-response-content
        nc_path_elements,           # 對應 nc-path-content
        summary_tools_elements,     # 2. 加在最後面 (對應 summary-tools-state)
    )


_stl_component_cache = None


@app.callback(
    Output("stl-graph", "figure"), 
    Input("stl-dropdown", "value"), 
    Input("workpiece-slider", "value")  # 1. 這裡改成新的 Slider ID
)
def update_stl_graph(stl_path, slider_value): # 2. 參數名稱跟著改，避免混淆
    current_idx = to_int(slider_value, 0) % len(CSV_SNAPSHOTS) if CSV_SNAPSHOTS else 0
    
    snapshot = {}
    if CSV_SNAPSHOTS:
        history = build_csv_window(CSV_SNAPSHOTS, current_idx)
        window_indices = compute_window_indices(CSV_SNAPSHOTS, current_idx)
        start = window_indices[0] if window_indices else current_idx
        now_pos = current_idx - start if window_indices else None
        
        # 確保有抓到當下的快照資料
        if now_pos is not None and 0 <= now_pos < len(history) and isinstance(history[now_pos], dict):
            snapshot = history[now_pos]
            
    # 將帶有 RA 數值的 snapshot 傳給繪圖函數，點點就會出現了
    return build_stl_figure(stl_path, snapshot)


if __name__ == "__main__":
    app.run(debug=True, port=8050)