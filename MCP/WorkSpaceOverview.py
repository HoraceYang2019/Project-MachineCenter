import json
import threading
from pathlib import Path
import yaml
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, dcc, html
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import FileResponse  # 新增這行：用來直接回傳實體檔案
import uvicorn
# 期末最終版
ROOT = Path(__file__).parent
DATA_FILE = ROOT / "CNC_opcua.json"
YAML_DIR = ROOT / "yaml_overview/index.html"
MQTT_TOPIC = "cnc/snapshot"
MQTT_BROKER = "localhost" 
MQTT_PORT = 1883

RaWARN_LEVEL = 0.5
RaDANGER_LEVEL = 0.8

Torque_WARN_LEVEL = 0.1
Torque_DANGER_LEVEL = 0.12

Bending_WARN_LEVEL = 2.0
Bending_DANGER_LEVEL = 2.6

Wear_WARN_LEVEL = 0.23
Wear_DANGER_LEVEL = 0.28

STATE_LOCK = threading.Lock()
LATEST_DATA = {}
LATEST_HISTORY = []
STL_CACHE = None

def build_empty_figure(message="沒有數值"):
    """產生一個只包含置中文字提示的空圖表"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20, color="#888", family="sans-serif")
    )
    fig.update_layout(
        xaxis=dict(visible=False),  # 隱藏 X 軸
        yaxis=dict(visible=False),  # 隱藏 Y 軸
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

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
    default_path = ROOT / "325BTM.STL"
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
            N = len(point_values)
            
            if N > 0:
                xmin = float(np.min(vertices[:, 0]))
                xmax = float(np.max(vertices[:, 0]))
                ymin = float(np.min(vertices[:, 1]))
                ymax = float(np.max(vertices[:, 1]))
                zmax = float(np.max(vertices[:, 2]))
                
                width = xmax - xmin
                y_center = (ymin + ymax) / 2  # Y 固定在模型正中間 (y=0 的概念)
                
                x_positions = []
                y_positions = []
                z_positions = []
                
                # 根據 N 個點，將寬度切成 N+1 等份
                for i in range(1, N + 1):
                    x_val = xmin + (i / (N + 1)) * width
                    x_positions.append(x_val)
                    y_positions.append(y_center)
                    # Z 軸稍微浮貼在模型上方
                    z_positions.append(zmax + max((zmax - float(np.min(vertices[:, 2]))) * 0.02, 0.01))
                
                marker_colors = [point_color_for_ra(value) for value in point_values]
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

def to_float(value, default=0.0):
    """
    將傳入的 value 安全地轉換為 float。
    若 value 為 None 或無法轉換的字串（如 "-"、"N/A" 等），則回傳預設值 default。
    """
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

# 1. 建立一個全域字典，將 sub_num 1 與 2 分開存放
REAL_DATA_CACHE = {
    "1": [],
    "2": []
}

BASE_DIR = ROOT / "承威data" / "step1_ucl_pipeline_2" / "data" / "exp"

def load_real_experiment_data():
    REAL_DATA_CACHE["1"].clear()
    REAL_DATA_CACHE["2"].clear()
    
    for exp_num in range(29, 35):
        for sub_num in range(1, 3):
            folder_name = f"EXP-{exp_num}-{sub_num}"
            folder_path = BASE_DIR / folder_name
            
            if not folder_path.exists():
                continue # 若資料夾不存在則跳過
                
            # 1. 搜尋資料夾底下副檔名為 .yaml 或 .yml 的所有檔案
            yaml_files = list(folder_path.glob("*.yaml")) + list(folder_path.glob("*.yml"))
            
            if not yaml_files:
                print(f"提示: 資料夾 {folder_name} 內找不到任何 YAML 檔案。")
                continue
                
            # 2. 取得唯一的 YAML 檔案路徑
            target_yaml_path = yaml_files[0]
            
            try:
                # 3. 開啟並安全載入 YAML 檔案
                with open(target_yaml_path, "r", encoding="utf-8") as f:
                    yaml_content = yaml.safe_load(f)
                
                if not yaml_content:
                    continue
                    
                # 假設 yaml_content 已經由 yaml.safe_load(f) 成功讀取

                # 1. 安全地逐層往下挖取，直到取得 Summary 字典
                tool_related = yaml_content.get("ToolRelated", {})
                tool_holder = tool_related.get("ToolHolder", {})
                sth1 = tool_holder.get("STH1", {})
                summary = sth1.get("Summary", {})

                if not summary:
                    print(f"提示: {folder_name} 的 YAML 中找不到 Summary 資料。")
                    # 可以決定要 continue 跳過，或是給予預設值

                # 2. 從 Summary 中精準取出你要的數值
                max_torque = summary.get("MaxTorque", 0)  # 找不到時預設給 0.5
                max_bending = summary.get("MaxBending", 0)
                rms_torque = summary.get("rmsTorque", 0.0)  
                std_torque = summary.get("stdTorque", 0.0)

                # 3. 再往下挖取 VisualizationLinks 裡面的 HTML 檔名
                vis_links = summary.get("VisualizationLinks", {})
                torque_value_html = vis_links.get("Torque_Value_HTML", "")
                # 讀取Ra欄位
                # 假設你已經取得了 yaml_content
                job = yaml_content.get("Job", {})
                quality = job.get("Quality", {})
                surface_roughness = quality.get("SurfaceRoughness", {})

                # 1. 提取 Tolerance (找不到時預設給 0.8)
                tolerance = to_float(surface_roughness.get("Tolerance", 0.8), 0.8)

                # 2. 提取 Measurement 列表 (找不到時給空陣列 [])
                raw_measurements = surface_roughness.get("Measurement", [])
                
                prediction_ra = surface_roughness.get("Prediction", [])
                # 刀具磨耗區
                toolhouse = tool_related.get("ToolHouse", {})
                t1 = toolhouse.get("T1", {})
                actual_wear = t1.get("ActualWear", {})
                after = actual_wear.get("After", 0.0)
                
                # ==========================================
                predict = t1.get("Predict", {})
                predicted_wear = predict.get("Predict", 0.0)
                
                # 防呆機制：如果 YAML 裡面寫錯格式，確保它真的是一個 list
                if not isinstance(raw_measurements, list):
                    raw_measurements = []

                # 3. 透過串列生成式，將陣列裡的每一個值安全地轉成 float
                ra_list = [to_float(m, 0.0) for m in raw_measurements if m is not None]
                
                # 4. 寫入你的 snapshot 準備給 Dash 畫圖使用
                snapshot = {
                    "workpiece_id": f"{exp_num}-{sub_num}",
                    "Toque": to_float(max_torque, 0.0),      # 對應 Dashboard 上的 Torque
                    "Bending": to_float(max_bending, 0.0),  # 對應 Dashboard 上的 Bending
                    "rmsTorque": to_float(rms_torque, 0.0),
                    "Torque_Value_HTML": torque_value_html,      # 把 Iframe 需要的路徑也存起來
                    "Ra_List": ra_list,
                    "Tolerance": tolerance,
                    "Prediction_Ra": prediction_ra,
                    "Tool_Life": to_float(after, 0.0),
                    "Predicted_Tool_Life": to_float(predicted_wear, 0.0)
                }
                
                # 5. 依據 sub_num (1 或 2) 存入對應的快取分組
                REAL_DATA_CACHE[str(sub_num)].append(snapshot)
                
            except Exception as error:
                print(f"解析檔案 {target_yaml_path.name} 時發生異常: {error}")

# 啟動時執行一次讀取
load_real_experiment_data()



# Ra盒狀圖
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
    
    # 如果有數據，就用索引去查出真正的 workpiece_id；查不到才用數字代替
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
    # [修正] 既然現在資料已經是乾淨的 List，直接拿出來即可
    return snapshot.get("Ra_List", [])


def to_float(value, default=0.0):
    """
    將傳入的 value 安全地轉換為 float。
    允許 default 傳入 None，以配合前端的「無數值 (Empty State)」判定。
    """
    if value is None or str(value).strip() == "" or value == "-":
        return default if default is None else float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default if default is None else float(default)


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


def build_history_trend_figure(title, values, warn_level, danger_level, labels=None, now_pos=None, x_values=None, pred_values=None):
    if labels is None:
        labels = [str(i) for i in range(len(values))]
    if x_values is None:
        x_values = list(range(len(values)))
    
    figure = go.Figure()

    # === 第 1 層：預測值 (灰色虛線，放底層) ===
    if pred_values:
        plot_len = min(len(x_values), len(pred_values))
        figure.add_trace(
            go.Scatter(
                x=x_values[:plot_len],
                y=pred_values[:plot_len],
                mode="lines+markers",
                line=dict(color="#95a5a6", width=2, dash="dash"),
                marker=dict(color="#95a5a6", size=6),
                name="Pred",         # 圖例顯示名稱
                hoverinfo="y+name"
            )
        )

    # === 第 2 層：實際值 (藍色實線，一條畫到底) ===
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=values,
            mode="lines+markers+text",
            text=[f"{v:.2f}" for v in values],
            textposition="top center",
            line=dict(color="#1f77b4", width=3),
            marker=dict(color="#1f77b4", size=8),
            name="Real"             # 圖例顯示名稱
        )
    )

    # === 第 3 層：標示當前位置 NOW ===
    if now_pos is not None and 0 <= now_pos < len(x_values):
        # 1. 在實際值的線上疊加一個明顯的大綠點
        figure.add_trace(
            go.Scatter(
                x=[x_values[now_pos]],
                y=[values[now_pos]],
                mode="markers",
                marker=dict(color="#2ecc71", size=14, symbol="circle", line=dict(width=2, color="white")),
                name="NOW",
                showlegend=False,      # 隱藏這個點的圖例，保持乾淨
                hoverinfo="skip" 
            )
        )
        
        # 2. 畫一條垂直的綠色輔助線與文字
        selected_x = x_values[now_pos]
        figure.add_vline(x=selected_x, line_color="#2ecc71", line_width=3, line_dash="solid")
        figure.add_annotation(
            x=selected_x, y=1.02, xref="x", yref="paper", 
            text="NOW", showarrow=False, font=dict(color="#2ecc71", size=12)
        )

    # === 第 4 層：加入警戒線與危險線 ===
    figure.add_hline(y=warn_level, line_dash="dot", line_color="#f5a623")
    figure.add_hline(y=danger_level, line_dash="dot", line_color="#d0021b")
    
    # === 排版 Know-how：開啟圖例並平鋪於右上角 ===
    figure.update_layout(
        title=title,
        margin=dict(l=16, r=16, t=42, b=16),
        height=230,
        autosize=False,
        showlegend=True,         # 強制開啟圖例
        legend=dict(
            orientation="h",     # 水平排列
            yanchor="bottom", 
            y=1.02,              # 貼齊圖表頂端
            xanchor="right", 
            x=1                  # 靠右對齊
        ),
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
                # 統計該批次 6 個點的 Ra 值落在哪個區間
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
        title="Ra Status",
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


def build_history_box_figure(history, labels=None, now_pos=None, tolerance=None):
    if not history:
        return build_empty_figure("無歷史數據")

    fig = go.Figure()
    
    # 建立兩組獨立的儲存箱
    x_vals_actual = []
    y_vals_actual = []
    
    x_vals_pred = []
    y_vals_pred = []
    
    # 把歷史資料中的陣列攤平 (Flatten) 給 Plotly 畫箱型圖
    for snap in history:
        w_id = snap.get("workpiece_id", "Unknown")
        ra_list = snap.get("Ra_List", [])
        
        # 兼容大小寫，確保一定抓得到先前存入的預測陣列
        prediction_ra = snap.get("prediction_Ra", snap.get("Prediction_Ra", []))
        
        # 1. 獨立攤平：實際量測值
        for ra in ra_list:
            x_vals_actual.append(w_id)
            y_vals_actual.append(ra)
            
        # 2. 獨立攤平：預測值
        for p_ra in prediction_ra:
            x_vals_pred.append(w_id)
            y_vals_pred.append(p_ra)
            
    # === 新增：畫出 Prediction 預測值 ===
    if y_vals_pred:
        fig.add_trace(go.Box(
            x=x_vals_pred, 
            y=y_vals_pred,
            marker_color="#95a5a6", # 灰色
            name="Pred"
        ))
        
    # === 原本：畫出實際 Ra 分布 ===
    if y_vals_actual:
        fig.add_trace(go.Box(
            x=x_vals_actual, 
            y=y_vals_actual,
            marker_color="#3182bd", # 藍色
            name="Actual"
        ))

    # 標示當前選擇的工件位置
    if now_pos is not None and now_pos < len(history):
        fig.add_vline(
            x=now_pos,                
            line_dash="dash",
            line_color="#2ecc71",
            line_width=2,
            annotation_text="NOW",
            annotation_position="top right",
            annotation_font=dict(color="#2ecc71", size=12)
        )

    # 畫上 Tolerance 虛線
    if tolerance is not None:
        fig.add_hline(
            y=tolerance, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f" ", 
            annotation_position="top left"
        )

    fig.update_layout(
        title="History",
        margin=dict(l=20, r=20, t=40, b=20),
        # 關鍵 Know-how: 設定 group，讓同一個工件的預測箱與實際箱並排，不會互相覆蓋
        boxmode='group' ,
        plot_bgcolor="#e6ecf5",   # 內部繪圖區的背景顏色 (例如：淺灰藍色)
        # === 關鍵 Know-how：強制設定 X 軸為純類別字串 ===
        xaxis=dict(
            type="category"
        )
    )
    return fig


def build_ra_histogram_figure(base_value, batch_offset):
    points = [max(0.0, base_value + batch_offset * 0.01 + (index - 10) * 0.02 + (index % 4) * 0.005) for index in range(30)]
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=points, nbinsx=10, marker_color="#1f77b4"))
    figure.add_vline(x=base_value + 0.8, line_dash="dash", line_color="#f5a623")
    figure.add_vline(x=base_value + 1.5, line_dash="dash", line_color="#d0021b")
    figure.update_layout(title="Ra Histogram", margin=dict(l=16, r=16, t=42, b=16), height=230, autosize=False, showlegend=False, xaxis_title="推論品質", yaxis_title="數量")
    return figure


import plotly.graph_objects as go

def build_history_ra_histogram_figure(snapshot):
    # 1. 取得實際陣列、預測陣列與公差
    ra_list = snapshot.get("Ra_List", [])
    prediction_ra = snapshot.get("Prediction_Ra", [])
    tolerance = snapshot.get("Tolerance", 0.8)
    
    if not ra_list:
        return build_empty_figure("無 Ra 量測數據")

    # 2. 建立 X 軸標籤 (測點 1, 測點 2...)
    x_labels = [f"P{i+1}" for i in range(len(ra_list))]
    
    # 3. 判斷每個實際點是否超標
    marker_colors = ["#d0021b" if val > tolerance else "#1f77b4" for val in ra_list]

    fig = go.Figure()
    
    # === 新增：畫出 Prediction 預測值 (放在底層) ===
    if prediction_ra:
        fig.add_trace(go.Scatter(
            x=x_labels[:len(prediction_ra)], # 確保長度不超過 X 軸標籤
            y=prediction_ra,
            mode="lines+markers",
            line=dict(color="#95a5a6", width=2, dash="dash"), # 灰色虛線
            marker=dict(color="#95a5a6", size=6),
            name="Pred",
            hoverinfo="y+name"
        ))

    # === 原本的：畫出 Actual 實際量測值 (放在上層) ===
    fig.add_trace(go.Scatter(
        x=x_labels, 
        y=ra_list,
        mode="lines+markers+text",
        text=[f"{val:.3f}" for val in ra_list], 
        textposition="top center",
        line=dict(color="#1f77b4", width=2),    
        marker=dict(color=marker_colors, size=10, line=dict(width=1, color="white")), 
        name="Actual"
    ))
    
    # 畫一條水平的公差基準線
    fig.add_hline(
        y=tolerance, 
        line_dash="dash", 
        line_color="red", 
        annotation_text=f" ", 
        annotation_position="top left"
    )

    fig.update_layout(
        title="Current",
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(title="Ra", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), # 將圖例放在右上角
        plot_bgcolor="#e6ecf5",   # 內部繪圖區的背景顏色 (例如：淺灰藍色)
    )
    return fig


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


# 建立滑桿標記字典
slider_marks = {}


MQTT_CLIENT = connect_mqtt()


#  建立 Dash 應用程式
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder=str(ROOT / "assets"),requests_pathname_prefix='/workspace/')

HTML_BASE_DIR = ROOT / "承威data" / "step1_ucl_pipeline_2"
#  建立 FastAPI 主伺服器
fastapi_app = FastAPI()

# 1. 掛載靜態資源 (這段你原本已經有了)
fastapi_app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")

fastapi_app.mount(
    "/raw_html", 
    StaticFiles(directory=HTML_BASE_DIR), 
    name="raw_html"
)

@fastapi_app.get("/")
def serve_home_page():
    html_path = str(ROOT / "assets" / "index.html") 
    return FileResponse(html_path)
 
# 2. 【關鍵修正】把 Dash 的 WSGI 伺服器掛載到 /workspace 之下
fastapi_app.mount("/workspace/", WSGIMiddleware(app.server))

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H2("Analysis"), md=2),
                dbc.Col(# --- 新增/修改：Sub Num 切換開關與滑桿 ---
                                html.Div(style={"marginTop": "12px", "padding": "0 16px"}, children=[
                                    # 2. 原本的滑桿 (此處的 marks, max, value 稍後會由 Callback 動態接管)
                                    dbc.Row([   
                                        dbc.Col([
                                            dcc.Slider(
                                                id="workpiece-slider",
                                                min=0,
                                                max=1,
                                                step=1,
                                                value=0,
                                            )
                                        ], md=10),
                                    ], className="mb-3 justify-content-center"),
                                ]),
                        md=7),
                dbc.Col(dbc.Button("Plan", 
                                   id="return-btn", 
                                   color="light", 
                                   className="w-100", 
                                   href="/",
                                   external_link=True,
                                   style={"text-decoration": "none", "color": "#1a73e8", "font-weight": "bold", "font-size": "1.1rem", "padding": "8px 12px", "border-radius": "6px", "border": "1px solid #1a73e8", "transition": "all 0.2s"}     
                                   ), md=1),
                dbc.Col(dbc.Button("Monitor", 
                                   id="return-btn", 
                                   color="light", 
                                   className="w-100", 
                                   href="http://127.0.0.1:8051",
                                   external_link=True,
                                   style={"text-decoration": "none", "color": "#1a73e8", "font-weight": "bold", "font-size": "1.1rem", "padding": "8px 12px", "border-radius": "6px", "border": "1px solid #1a73e8", "transition": "all 0.2s"}              
                                   ), md=1),
                dbc.Col(dbc.Button("Replay", 
                                   id="return-btn", 
                                   color="light", 
                                   className="w-100", 
                                   href="http://127.0.0.1:8055",
                                   external_link=True,
                                   style={"text-decoration": "none", "color": "#1a73e8", "font-weight": "bold", "font-size": "1.1rem", "padding": "8px 12px", "border-radius": "6px", "border": "1px solid #1a73e8", "transition": "all 0.2s"}              
                                   ), md=1),
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
                # 左側欄位
                html.Div(
                    [
                        # ---------------------------------------------------------
                        # 1. 產線區塊 (上方)
                        # --------------------------------------------------------- 
                        dbc.Accordion(
                            id="accordion-tool-operator",   # <--- 新增 Accordion ID
                            children=[
                                dbc.AccordionItem(
                                item_id="item-tool-op", # <--- 新增抽屜身分證
                                title=html.Div([
                                # 載入 assets 資料夾內的 png，並透過 style 限制大小與對齊
                                    html.Img(
                                    src="/assets/pngtree-product-production-line-icon-png-image.png", 
                                    style={
                                        "width": "20px",           # 限制寬度
                                        "height": "20px",          # 限制高度
                                        "marginRight": "8px",      # 與右側文字保持間距
                                        "verticalAlign": "middle"  # 確保圖片與文字垂直置中對齊
                                    }
                                ),html.Span("Tool Wear Analytics & Responses", style={"verticalAlign": "middle"})
                                ],style={"display": "inline-block"}),
                                children=[    
                                          
                                    dcc.Graph(id="wear-figure", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),

                                    # --- 新增：Tool States by TID ---
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.H6("Tool States by TID", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                                            html.Div(id="tool-states-content", style={"fontSize": "14px", "minHeight": "60px"})
                                        ]),
                                    ),
                                    
                                    # --- 新增：Next Responses for Tool Issues ---
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.H6("Next Responses for Tool Issues", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                                            html.Div(id="tool-response-content", style={"fontSize": "14px", "minHeight": "60px"})
                                        ]),
                                    ),
                                ]
                                )    
                            ],
                            start_collapsed=True,
                            className="mt-2",
                        ),
                        # ---------------------------------------------------------
                        # 2. 工程師區塊 (下方)
                        # ---------------------------------------------------------
                        dbc.Accordion(
                            id="accordion-tool-engineer",   # <--- 新增 Accordion ID
                            children=[
                                dbc.AccordionItem(
                                    item_id="item-tool-eng", # <--- 新增抽屜身分證
                                    title=html.Div([
                                    html.Img( # 載入 assets 資料夾內的 png，並透過 style 限制大小與對齊
                                    src="/assets/equipment-engineer-icon-png-image.png", 
                                    style={
                                        "width": "20px",           # 限制寬度
                                        "height": "20px",          # 限制高度
                                        "marginRight": "8px",      # 與右側文字保持間距
                                        "verticalAlign": "middle"  # 確保圖片與文字垂直置中對齊
                                    }
                                ),html.Span("Cutting Forces", style={"verticalAlign": "middle"})
                                ],style={"display": "inline-block"}),
                                    children=[
                                        dcc.Graph(id="toque-figure", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                                        dcc.Graph(id="bending-figure", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}), 
                                    ],
                                )
                            ],
                            start_collapsed=True,
                            className="mt-2",

                        ),
           
                    ],
                    className="left-panel",

                ),
                # 
                html.Div(
                    [
                        # --- 替換：States Summary 頂部狀態列 ---
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "padding": "10px 16px",
                                "backgroundColor": "#f8f9fa",
                                "border": "1px solid #dee2e6",
                                "borderRadius": "4EXpx",
                                "marginBottom": "0px",
                            },
                            children=[
                                # 左側：Target Tool 與 動態燈號按鈕
                                html.Div(
                                    style={"flex": "1", "display": "flex", "alignItems": "center"},
                                    children=[
                                        html.Span("Target Tool:", style={"fontWeight": "bold", "marginRight": "12px", "color": "#666"}),
                                        dcc.RadioItems(
                                            id="sub-num-selector",
                                            value="1",     # 預設選中 T1
                                            inline=True,
                                            # 隱藏預設圓形單選按鈕
                                            inputStyle={"display": "none"}, 
                                            labelStyle={"cursor": "pointer", "marginRight": "12px", "display": "inline-flex", "alignItems": "center"}
                                        )
                                    ]
                                ),
                                
                                # 中間： 工件名稱
                                html.Div(
                                    children=[
                                        html.Span(style={"fontWeight": "bold", "fontSize": "16px", "color": "#333"}),
                                        html.Span(id="summary-workpiece-name", style={"fontWeight": "bold", "fontSize": "16px", "color": "#1f77b4"})
                                    ],
                                    style={"textAlign": "center", "flex": "1"}
                                ),
                                
                                # 右側：HTML 與 STL 視圖切換
                                
                                html.Div(
                                    style={"flex": "1", "textAlign": "right"},
                                    children=[
                                        dcc.RadioItems(
                                            id="view-mode-selector",
                                            options=[
                                                {"label": " Engineer View ", "value": "html"},
                                                {"label": " Operator View ", "value": "stl"}
                                            ],
                                            value="html",
                                            inline=True,
                                            inputStyle={"marginRight": "4px", "marginLeft": "12px"}
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            style={"display": "flex", "flexDirection": "column", "alignItems": "stretch", "flex": "1 1 auto", "minHeight": "0", "overflow": "hidden"},
                            children=[
                                html.Div(
                                    id="center-view-container",  # 名字必須跟 Callback 輸出的一模一樣
                                    style={"position": "relative", "width": "100%", "flex": "1 1 auto", "minHeight": "0", "height": "100%", "display": "flex", "justifyContent": "center", "alignItems": "center", "backgroundColor": "#f8f9fa", "border": "1px solid #dee2e6", "borderRadius": "4px", "overflow": "hidden"},
                                    children=[]  # 裡面必須完全清空！把 Iframe 拔掉
                                ),
                                
                                # --- 新增/替換：NC Path List ---
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H6("NC Path List", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                                        html.Div(id="nc-path-content", style={"fontSize": "14px", "minHeight": "40px", "fontFamily": "monospace", "backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "4px"})
                                    ]),
                                    className="mt-2", style={"boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
                                ),
                            ],
                        ),
                    ],
                    id="center-panel-container",
                    className="center-panel",
                ),
                # 右側欄位
                html.Div([   
                    # ---------------------------------------------------------
                    # 3. 產線區塊 (上方)
                    # ---------------------------------------------------------
                    dbc.Accordion(
                        id="accordion-quality-operator",   # <--- 新增 Accordion ID
                        children=[
                            dbc.AccordionItem(
                                item_id="item-quality-op", # <--- 新增抽屜身分證
                                title=html.Div([
                                html.Img(
                                    src="/assets/pngtree-product-production-line-icon-png-image.png", 
                                    style={
                                        "width": "20px",           # 限制寬度
                                        "height": "20px",          # 限制高度
                                        "marginRight": "8px",      # 與右側文字保持間距
                                        "verticalAlign": "middle"  # 確保圖片與文字垂直置中對齊
                                    }
                                ),html.Span("Quality Analytics & Responses", style={"verticalAlign": "middle"})
                                ],style={"display": "inline-block"}),
                                children=[
                                    
                                    dcc.Graph(id="status-heatmap", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                                    # 1. Quality States by WID
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.H6("Quality States by WID", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                                            html.Div(id="quality-states-content", style={"fontSize": "14px", "minHeight": "60px"})
                                        ]),
                                        className="mb-2", style={"boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
                                    ),
                                    
                                    # 2. Next Responses for Quality Issues
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.H6("Next Responses for Quality Issues", className="card-title", style={"fontWeight": "bold", "borderBottom": "1px solid #ccc", "paddingBottom": "4px"}),
                                            html.Div(id="quality-response-content", style={"fontSize": "14px", "minHeight": "60px"})
                                        ]),
                                        className="mb-2", style={"boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
                                    ),
                                    
                                ]
                            )
                        ],
                        start_collapsed=True, # 預設為收合狀態，設為 False 則預設為展開
                        className="mt-2",
                    ),
                    # ---------------------------------------------------------
                    # 4. 工程師區塊 (下方)
                    # ---------------------------------------------------------
                    # Quality Check
                    dbc.Accordion(
                        id="accordion-quality-engineer",   # <--- 新增 Accordion ID
                        children=[
                            dbc.AccordionItem(
                                item_id="item-quality-eng", # <--- 新增抽屜身分證
                                title=html.Div([
                                # 載入 assets 資料夾內的 png，並透過 style 限制大小與對齊
                                html.Img(
                                    src="/assets/equipment-engineer-icon-png-image.png", 
                                    style={
                                        "width": "20px",           # 限制寬度
                                        "height": "20px",          # 限制高度
                                        "marginRight": "8px",      # 與右側文字保持間距
                                        "verticalAlign": "middle"  # 確保圖片與文字垂直置中對齊
                                    }
                                ),html.Span("Quality Check", style={"verticalAlign": "middle"})
                                ],style={"display": "inline-block"}),
                                children=[
                                    # 3. Ra 趨勢圖表
                                    dcc.Graph(id="ra-box", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                                    dcc.Graph(
                                        id="ra-trend", 
                                        config={"displayModeBar": False, "responsive": True}, 
                                        style={"height": "240px"}
                                    ),
                                ]
                            )
                        ],
                        start_collapsed=True, # 預設為收合狀態，設為 False 則預設為展開
                    )
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
    style={
        "width": "100%",
        "height": "100vh",
        "maxWidth": "100%",
        "margin": "0",
        "padding": "0",
    },
)

@app.callback(
    Output("accordion-tool-operator", "active_item"),
    Output("accordion-tool-engineer", "active_item"),
    Output("accordion-quality-engineer", "active_item"),
    Output("accordion-quality-operator", "active_item"),
    Input("view-mode-selector", "value")
)
def toggle_accordions(view_mode):
    # 關鍵 Know-how：使用空字串 "" 來強制關閉面板，避免 None 被忽略
    if view_mode == "html":
        # 打開左側刀具，關閉右側品質
        return "", "item-tool-eng", "item-quality-eng", ""
        
    elif view_mode == "stl":
        # 關閉左側刀具，打開右側品質
        return "item-tool-op", "", "", "item-quality-op"
        
    # 預設防呆全關
    return "", "", "", ""

@app.callback(
    Output("workpiece-slider", "marks"),
    Output("workpiece-slider", "max"),
    Output("workpiece-slider", "value"),
    Input("sub-num-selector", "value"),
    State("workpiece-slider", "value")  # <--- 新增：偷偷記住滑桿現在停在哪個 index
)
def update_slider_for_sub_num(selected_sub, current_slider_value):
    # 根據選擇的 sub_num 取出對應的陣列
    data_list = REAL_DATA_CACHE.get(selected_sub, [])
    count = len(data_list)
    
    if count == 0:
        return {0: "No Data"}, 0, 0
        
    marks = {}
    for i, snap in enumerate(data_list):
        w_id = str(snap.get("workpiece_id", f"{i}"))
        ra_val = to_float(snap.get("Ra", snap.get("Offset_Z", 0.0)), 0.0)
        mark_color = point_color_for_ra(ra_val)
        
        marks[i] = {
            "label": w_id,
            "style": {
                "whiteSpace": "nowrap",  # <--- 關鍵 Know-how：強制文字絕對不換行
                "color": "black",  
                "fontWeight": "700",
                "fontSize": "12px",
                "textShadow": f"1px 1px 0 {mark_color}, -1px 1px 0 {mark_color}, -1px -1px 0 {mark_color}, 1px -1px 0 {mark_color}"
            }
        }
        
    new_max = count - 1
    
    # === Know-how: 狀態保留與防呆邏輯 ===
    # 1. 處理初始載入：如果是系統剛啟動 (None)，則預設停在 index 0
    safe_value = current_slider_value if current_slider_value is not None else 0
    
    # 2. 處理資料不對等：假設 T1 做了 30 顆，T2 只做了 15 顆
    # 如果滑桿原本停在 25，切換到 T2 時會超出陣列範圍報錯。
    # 所以必須將其限制在新的最大值 (new_max) 之內。
    if safe_value > new_max:
        safe_value = new_max
        
    # 將原本寫死的 0，換成經過防呆處理的 safe_value
    return marks, new_max, safe_value


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
    Output("nc-path-content", "children"),
    Output("sub-num-selector", "options"),       # 動態改變按鈕的顏色與外框    
    Output("center-view-container", "children"),
    Input("mqtt-refresh", "n_intervals"),
    Input("workpiece-slider", "value"),
    Input("sub-num-selector", "value"),  # 新增監聽開關
    Input("view-mode-selector", "value"),
)

def update_figures(_n_intervals, slider_value, selected_sub, view_mode):
    # 動態取得當前選定 Sub 的資料陣列
    current_dataset = REAL_DATA_CACHE.get(selected_sub, [])

    
    if current_dataset:
        current_idx = to_int(slider_value, 0)
        if current_idx >= len(current_dataset):
            current_idx = len(current_dataset) - 1
        snapshot = current_dataset[current_idx]
        
        window_indices = compute_window_indices(current_dataset, current_idx)
        window_labels = compute_window_labels(current_dataset, current_idx)
        
        # [修正] history 必須是「時間窗口」內的資料，這樣才同時包含過去與未來
        history = [current_dataset[i] for i in window_indices]
        
        # [修正] 計算 NOW 這個點在時間窗口裡面的索引位置
        now_pos = window_indices.index(current_idx) if current_idx in window_indices else 0
    else:
        # 找不到資料時的備用邏輯
        snapshot = {}
        history = []
        window_labels = None
        now_pos = None
        
    # 5. 更新 batch_offset 的定義，使其與當前查到的 workpiece_id 連動（供模擬數據增量使用）
    wear_base = to_float(snapshot.get("Tool_Life", None), None)
    pred_wear_values = to_float(snapshot.get("Predicted_Tool_Life", wear_base), wear_base) 
    toque_base = to_float(snapshot.get("Toque", snapshot.get("Radius_Comp", None)), None)
    bending_base = to_float(snapshot.get("Bending", snapshot.get("Length_Comp", None)), None)
    
    # [修正] 從 Ra_List 中取出最大的粗糙度作為這顆工件的代表值
    ra_list_temp = snapshot.get("Ra_List", [])
    ra_base = max(ra_list_temp) if ra_list_temp else 0.0
    tool_code = snapshot.get("Tool_Code", "-")
    status_text = str(snapshot.get("Status", "-")).upper()
    # === 新增：動態 Iframe 路徑邏輯 ===
    # 從快照中取出 HTML 檔名，若無則給予預設值防呆
    html_filename = snapshot.get("Torque_Value_HTML")
    
    # 1. 從快照取得 YAML 裡的檔案路徑
    html_filename = snapshot.get("Torque_Value_HTML", "")
    
    # 2. 洗乾淨斜線：將 Windows 的反斜線 \ 替換為網址用的正斜線 /
    clean_filepath = html_filename.replace("\\", "/")
    
    # 3. 組合出合法的 URL，並透過剛才建立的 Flask 通道讀取
    iframe_src_url = f"/raw_html/{clean_filepath}" if clean_filepath else ""
    
    wear_values = [to_float(item.get("Tool_Life", wear_base), wear_base) for item in history if isinstance(item, dict)]
    pred_wear_values = [to_float(item.get("Predicted_Tool_Life", wear_base), wear_base) for item in history if isinstance(item, dict)]
    toque_values = [to_float(item.get("Toque", toque_base), toque_base) for item in history if isinstance(item, dict)]
    bending_values = [to_float(item.get("Bending", bending_base), bending_base) for item in history if isinstance(item, dict)]
    x_positions = list(range(len(history)))
    # === 動態生成 Ra 測點清單 ===
    ra_list = snapshot.get("Ra_List", [])
    tolerance = snapshot.get("Tolerance", 0.8)
    
    # 建立一個用來裝 HTML 元件的空陣列
    ra_ui_elements = []
    
    if not ra_list:
        # 如果沒有量測資料的空狀態
        ra_ui_elements.append(
            html.Div("無 Ra 量測數據", style={"color": "#999", "fontStyle": "italic", "textAlign": "center"})
        )
    else:
        # 動態顯示每一個測點，順便加入 Tolerance 的防呆判斷
        ra_ui_elements.append(html.Div(f"公差標準 (Tolerance): {tolerance}", style={"fontSize": "12px", "color": "#666", "marginBottom": "8px"}))
        
        for i, ra_val in enumerate(ra_list):
            # 判斷這個測點是否超過公差
            is_out_of_tol = ra_val > tolerance
            
            # 設定顏色與圖示
            color = "#d0021b" if is_out_of_tol else "#28a745" # 紅色(超標) / 綠色(正常)
            icon = "❌" if is_out_of_tol else "✅"
            
            # 建立單一個測點的 UI
            point_ui = html.Div(
                f"測點 {i+1}: {ra_val:.3f} {icon}",
                style={"color": color, "fontWeight": "bold", "padding": "2px 0", "borderBottom": "1px dashed #eee"}
            )
            ra_ui_elements.append(point_ui)

    # 把組裝好的陣列包裝進一個大 Div 裡面，傳遞給前端
    quality_states_content = html.Div(ra_ui_elements)
    # 2. 判斷是否為無效數值 (None 或是字串無法轉為數字)
    try:
        # 嘗試轉換為浮點數，如果本來就是 None 或 '-' 會拋出 ValueError / TypeError
        val = float(wear_values[-1])  # 只檢查最新的數值是否有效
        is_valid_value = True
    except (TypeError, ValueError):
        is_valid_value = False

    # 3. 根據判斷結果決定要畫什麼圖
    if not is_valid_value:
        # 如果沒有數值，回傳帶有文字的空圖表
        wear_figure = build_empty_figure("目前沒有 Tool Wear 數值")
    else:
        wear_figure = build_history_trend_figure("Tool Wear", wear_values, Wear_WARN_LEVEL, Wear_DANGER_LEVEL, labels=window_labels, now_pos=now_pos, x_values=x_positions,pred_values=pred_wear_values)
    toque_figure = build_history_trend_figure("Max Torque (Nm)", toque_values, Torque_WARN_LEVEL, Torque_DANGER_LEVEL, labels=window_labels, now_pos=now_pos, x_values=x_positions)
    bending_figure = build_history_trend_figure("Max Bending (Nm)", bending_values, Bending_WARN_LEVEL, Bending_DANGER_LEVEL, labels=window_labels, now_pos=now_pos, x_values=x_positions)
    heatmap_figure = build_history_heatmap_figure(history, labels=window_labels, now_pos=now_pos)
    box_figure = build_history_box_figure(history, labels=window_labels, now_pos=now_pos, tolerance=tolerance)    
    ra_trend_figure = build_history_ra_histogram_figure(snapshot)
    gauge_figure = build_history_status_gauge_figure(history)
    wear_value = f"{wear_base:.2f}" if wear_base is not None else "-"    
    toque_value = f"{toque_base:.4f}" if toque_base is not None else "-"
    bending_value = f"{bending_base:.2f}" if bending_base is not None else "-"
    ra_value = f"{ra_base:.2f}"
    wear_subtitle = f"Tool_Code: {tool_code}"
    toque_subtitle = f"來源: {status_text}"
    bending_subtitle = "上下界"
    ra_subtitle = f"Status: {status_text}"
    if wear_base is None:
        wear_badge = "N/A"
        badge_color = "#999999"  # 灰色
    elif wear_base <= 0.25:
        wear_badge = "OK"
        badge_color = "#28a745"  # 綠色
    else:
        wear_badge = "WARN"
        badge_color = "#dc3545"  # 紅色
    toque_badge = "-" if toque_base is None else ("OK" if toque_base <= Torque_DANGER_LEVEL else "WARN")
    bending_badge = "-" if bending_base is None else ("OK" if bending_base <= Bending_DANGER_LEVEL else "WARN")
    ra_badge = "OK" if ra_base <= RaDANGER_LEVEL else "WARN"
    raw_json = html.Pre(json.dumps(snapshot, indent=2, ensure_ascii=False), className="raw-json d-none")
    # === 新增：Tool States by TID 邏輯 ===
    tool_state_elements = [html.Div(f"Current Tool: T{tool_code}")]
    if wear_base is None or toque_base is None:
        tool_state_elements.append(html.Div("State: Data Unavailable", style={"color": "#999999", "fontWeight": "bold"}))
    elif bending_base > Bending_DANGER_LEVEL:
        tool_state_elements.append(html.Div("State: Bending Overload DANGER", style={"color": "#d0021b", "fontWeight": "bold"}))
    elif toque_base > Torque_WARN_LEVEL:
        tool_state_elements.append(html.Div("State:Torque Overload Warning", style={"color": "#f5a623", "fontWeight": "bold"}))
    else:
        tool_state_elements.append(html.Div("State: Normal", style={"color": "#43A047"}))

    # === 新增：Next Responses for Tool Issues 邏輯 ===
    response_elements = []
    if wear_base is None or toque_base is None:
                tool_state_elements.append(html.Div("State: Data Unavailable", style={"color": "#999999", "fontWeight": "bold"}))
    elif wear_base > Wear_WARN_LEVEL:
        response_elements.append(html.Div("1. 磨耗過大，刀具壽命即將到達極限，請準備更換備品。"))
        response_elements.append(html.Div("2. 建議降低下一工件之進給率 (Feed Rate)。"))
    elif toque_base > Torque_DANGER_LEVEL:
        response_elements.append(html.Div("1. 主軸負載異常，請檢查切削液供應與排屑狀態。"))
    else:
        response_elements.append(html.Div("維持原 NC 程式參數進行加工。", style={"color": "#666"}))
    # === 新增：Quality States by WID 邏輯 ===
    # 擷取工件名稱 (加入防呆機制：若 current_idx 未定義，則給予預設值 0)
    safe_idx = current_idx if 'current_idx' in locals() else 0
    current_wid = str(snapshot.get("workpiece_id", safe_idx))
    pts = extract_ra_points(snapshot)
    # 取出該工件 6 個點中的最大 Ra 值作為品質評級基準
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
    # 1. 優先排除異常狀態 (把 None 擋在第一關)
    if wear_base is None or toque_base is None:
        tool_status_color = "#999999"  # 資料異常(灰色)

    # 2. 判斷危險 (因為危險的條件比警戒嚴格)
    elif wear_base >= Wear_DANGER_LEVEL or toque_base >= Torque_DANGER_LEVEL:
        tool_status_color = "#d0021b"  # 危險(紅色)

    # 3. 判斷警戒 (如果沒進危險，)
    elif wear_base > Wear_WARN_LEVEL or toque_base > Torque_WARN_LEVEL:
        tool_status_color = "#f5a623"  # 警戒(橘色)


    # === 新增：動態計算 Sub 1 與 Sub 2 的即時狀態顏色 ===
    # === 1. 定義計算刀具顏色的 Know-how 邏輯 ===
    def get_tool_color(sub_id, idx):
        dataset = REAL_DATA_CACHE.get(sub_id, [])
        if not dataset or idx >= len(dataset):
            return "#999999"  # 無資料(灰色)
            
        snap = dataset[idx]
        wear = to_float(snap.get("Tool_Life", None), None)
        toque = to_float(snap.get("Toque", snap.get("Radius_Comp", None)), None)
        
        if wear is None or toque is None: return "#999999"
        if wear > Wear_DANGER_LEVEL: return "#d0021b" # 危險(紅色)
        if wear > Wear_WARN_LEVEL: return "#f5a623"    # 警戒(橘色)
        return "#3B963F" # 正常(綠色)

    # 取得當下這顆工件 (current_idx) 時，兩把刀的狀態燈號
    color_1 = get_tool_color("1", current_idx)
    color_2 = get_tool_color("2", current_idx)

    # === 2. 將燈號與按鈕 UI 結合 ===
    sub_num_options = [
        {
            "label": html.Span(
                "T1",
                style={
                    "backgroundColor": color_1,
                    "color": "white",
                    "padding": "4px 16px",
                    "borderRadius": "12px",
                    "fontSize": "13px",
                    "fontWeight": "bold",
                    # 如果是被選中的狀態，給予黑色粗外框與陰影提示
                    "border": "2px solid #333" if selected_sub == "1" else "2px solid transparent",
                    "boxShadow": "0px 2px 4px rgba(0,0,0,0.3)" if selected_sub == "1" else "none",
                }
            ),
            "value": "1"
        },
        {
            "label": html.Span(
                "T2",
                style={
                    "backgroundColor": color_2,
                    "color": "white",
                    "padding": "4px 16px",
                    "borderRadius": "12px",
                    "fontSize": "13px",
                    "fontWeight": "bold",
                    "border": "2px solid #333" if selected_sub == "2" else "2px solid transparent",
                    "boxShadow": "0px 2px 4px rgba(0,0,0,0.3)" if selected_sub == "2" else "none",
                }
            ),
            "value": "2"
        }
    ]

    # 取得當前時間點下，兩把刀各自的健康狀態
    color_1 = get_tool_color("1", current_idx)
    color_2 = get_tool_color("2", current_idx)

    # === 新增：組合 RadioItems 的選項 (動態結合顏色與選取外框) ===
    sub_num_options = [
        {
            "label": html.Div(
                "T1",
                style={
                    "backgroundColor": color_1,
                    "color": "white",
                    "padding": "6px 14px",
                    "borderRadius": "16px",
                    "fontSize": "13px",
                    "fontWeight": "bold",
                    # 實作的 Know-how: 判斷當前選取的是否為自己，是的話加上黑色粗外框與陰影
                    "border": "3px solid #333" if selected_sub == "1" else "3px solid transparent",
                    "boxShadow": "0px 2px 4px rgba(0,0,0,0.3)" if selected_sub == "1" else "none",
                    "transition": "all 0.2s ease" # 加上平滑動畫
                }
            ),
            "value": "1"
        },
        {
            "label": html.Div(
                "T2",
                style={
                    "backgroundColor": color_2,
                    "color": "white",
                    "padding": "6px 14px",
                    "borderRadius": "16px",
                    "fontSize": "13px",
                    "fontWeight": "bold",
                    "border": "3px solid #333" if selected_sub == "2" else "3px solid transparent",
                    "boxShadow": "0px 2px 4px rgba(0,0,0,0.3)" if selected_sub == "2" else "none",
                    "transition": "all 0.2s ease"
                }
            ),
            "value": "2"
        }
    ]
    
# 擷取工件名稱 (加入防呆機制：若 current_idx 未定義，則給予預設值 0)
    safe_idx = current_idx if 'current_idx' in locals() else 0
    current_wid = str(snapshot.get("workpiece_id", safe_idx))

    # # 2. 擷取 STL 檔名 (透過 pathlib 取得純檔名)
    # if stl_path:
    #     stl_name = Path(stl_path).name
    # elif STL_OPTIONS: # 預設載入第一個檔案的名稱
    #     stl_name = STL_OPTIONS[0]["label"]
    # else:
    #     stl_name = "未選擇 STL"

    # 3. 組合顯示字串
    center_display_text = f"325BTM - {current_wid}"
    # 判斷是否因 Torque 超標造成異常負載
    if toque_base is not None and toque_base > Torque_DANGER_LEVEL:        
        nc_path_elements.append(html.Div("[警告] 異常負載發生於:", style={"color": "#d0021b", "fontWeight": "bold"}))
        # 這裡放入模擬或實際抓取的 NC G-code
        nc_path_elements.append(html.Div("N0140 G01 X15.5 Y20.0 Z-5.0 F150", style={"color": "#d0021b"}))
        nc_path_elements.append(html.Div(f"原因分析: Torque 數值 ({toque_base:.2f} Nm) 超出警戒線 (40 Nm)", style={"color": "#666", "fontSize": "12px", "marginTop": "4px"}))
        
    # 判斷是否因 Bending 超標造成異常受力
    elif bending_base is not None and toque_base > Torque_WARN_LEVEL:        
        nc_path_elements.append(html.Div("[警告] 異常Toque受力發生於:", style={"color": "#f5a623", "fontWeight": "bold"}))
        nc_path_elements.append(html.Div("N0160 G02 X20.0 Y25.0 I4.5 J0.0 F120", style={"color": "#f5a623"}))
        nc_path_elements.append(html.Div(f"原因分析: Bending 數值 ({bending_base:.0f}) 超過正常上下界限", style={"color": "#666", "fontSize": "12px", "marginTop": "4px"}))
        
    # 若皆正常
    else:
        nc_path_elements.append(html.Div("目前未偵測到異常負載的 NC 段落。", style={"color": "#43A047"}))
        
    # === 新增：判斷要顯示 HTML 還是 STL ===
    if view_mode == "html":
        html_filename = snapshot.get("Torque_Value_HTML", "")
        clean_filepath = html_filename.replace("\\", "/")
        actual_file_path = HTML_BASE_DIR / clean_filepath if clean_filepath else None
        iframe_src_url = f"/raw_html/{clean_filepath}" if clean_filepath else ""
        
        center_view_content = html.Iframe(
            src=iframe_src_url,
            style={"width": "100%", "height": "100%", "border": "none", "display": "block"}
        )
    else:
        # 顯示 STL 模型的邏輯
        # 呼叫 build_stl_figure，並將目前的 snapshot 傳入以畫出 Ra 點
        stl_figure = build_stl_figure(measurement_snapshot=snapshot)
        center_view_content = dcc.Graph(
            figure=stl_figure,
            config={"displayModeBar": False, "responsive": False},
            style={"width": "100%", "height": "100%", "minHeight": "0"}
        )
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
        sub_num_options,            # 2. 加在最後面 
        center_view_content,             # 對應 stl-iframe 的 src
    )

if __name__ == "__main__":
    # 改由 uvicorn 啟動 FastAPI 主程式
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8052)
    