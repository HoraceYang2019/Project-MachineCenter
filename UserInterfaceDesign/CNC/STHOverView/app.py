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


ROOT = Path(__file__).parent
DATA_FILE = ROOT / "CNC_opcua.json"
CSV_FILE = ROOT / "machining_6stage.csv"
MQTT_TOPIC = "cnc/snapshot"
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

STATE_LOCK = threading.Lock()
LATEST_DATA = {}
LATEST_HISTORY = []
STL_CACHE = None


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
    # 以絕對位置顯示索引（與滑桿一致），避免再混用 I-3 / NOW 這種相對標示
    inds = compute_window_indices(snapshots, current_index)
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


def infer_ra_points(snapshot, point_count=6):
    points = []
    for i in range(1, point_count + 1):
        key = f"RA_P{i}"
        if key in snapshot:
            points.append(to_float(snapshot.get(key), 0.8))
    if len(points) == point_count:
        return points

    base_ra = to_float(snapshot.get("RA", snapshot.get("Offset_Z", 0.8)), 0.8)
    # Fallback: synthesize six inferred points around current RA value.
    return [max(0.0, base_ra + (i - 2.5) * 0.05 + (i % 2) * 0.01) for i in range(point_count)]


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
        xaxis=dict(tickmode="array", tickvals=x_values, ticktext=labels),
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
    if now_pos is not None and 0 <= now_pos < len(x_values):
        if now_pos > 0:
            figure.add_trace(
                go.Scatter(
                    x=x_values[:now_pos],
                    y=values[:now_pos],
                    mode="lines+markers",
                    line=dict(color="#1f77b4", width=3),
                    marker=dict(color="#1f77b4", size=8),
                    name="已加工",
                )
            )
        figure.add_trace(
            go.Scatter(
                x=[x_values[now_pos]],
                y=[values[now_pos]],
                mode="markers",
                marker=dict(color="#2ecc71", size=14, symbol="circle"),
                name="NOW",
            )
        )
        if now_pos + 1 < len(x_values):
            figure.add_trace(
                go.Scatter(
                    x=x_values[now_pos + 1 :],
                    y=values[now_pos + 1 :],
                    mode="lines+markers",
                    line=dict(color="#f39c12", width=3, dash="dash"),
                    marker=dict(color="#f39c12", size=8),
                    name="推測未來",
                )
            )
    else:
        figure.add_trace(
            go.Scatter(
                x=x_values[:split_at],
                y=values[:split_at],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=3),
                marker=dict(color="#1f77b4", size=8),
                name="過去 / 現在",
            )
        )
        if split_at < len(x_values):
            figure.add_trace(
                go.Scatter(
                    x=x_values[split_at:],
                    y=values[split_at:],
                    mode="lines+markers",
                    line=dict(color="#d62728", width=3, dash="dash"),
                    marker=dict(color="#d62728", size=8),
                    name="未來",
                )
            )
    if now_pos is not None and 0 <= now_pos < len(x_values) and now_pos < len(labels):
        selected_x = x_values[now_pos]
        figure.add_vline(x=selected_x, line_color="#2ecc71", line_width=3, line_dash="solid")
        figure.add_annotation(x=selected_x, y=1.02, xref="x", yref="paper", text="NOW", showarrow=False, font=dict(color="#2ecc71", size=12))
        if now_pos > 0:
            figure.add_annotation(x=x_values[0], y=1.02, xref="x", yref="paper", text="已加工", showarrow=False, font=dict(color="#1f77b4", size=12))
        if now_pos + 1 < len(x_values):
            figure.add_annotation(x=x_values[-1], y=1.02, xref="x", yref="paper", text="推測未來", showarrow=False, font=dict(color="#f39c12", size=12))
    figure.add_hline(y=warn_level, line_dash="dot", line_color="#f5a623")
    figure.add_hline(y=danger_level, line_dash="dot", line_color="#d0021b")
    figure.update_layout(
        title=title,
        margin=dict(l=16, r=16, t=42, b=16),
        height=230,
        autosize=False,
        showlegend=False,
        xaxis=dict(tickmode="array", tickvals=x_values, ticktext=labels),
        yaxis_title="數值",
    )
    return figure


def build_heatmap_figure(batch_offset , labels=None):
    labels = [str(i) for i in range(6)] if labels is None else labels
    categories = ["斷刀風險", "重刀", "正常", "輕刀"]
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
        labels=dict(x="工件視窗", y="類別", color="機率"),
    )
    figure.update_layout(title="狀態分布圖", margin=dict(l=16, r=16, t=42, b=16), height=230, autosize=False)
    return figure


def build_history_heatmap_figure(history, labels=None, now_pos=None):
    if labels is None:
        labels = [str(i) for i in range(6)]
    categories = ["斷刀風險", "重刀", "正常", "輕刀"]
    if not history:
        return build_heatmap_figure(0)

    palette = {
        "斷刀風險": "#e74c3c",
        "重刀": "#f1c40f",
        "正常": "#2ecc71",
        "輕刀": "#2d7ff9",
    }
    # 以 history 的位置順序產生矩陣；缺資料位置以正常分布作為中性值
    matrix = []
    for category in categories:
        row = []
        for snapshot in history:
            if not isinstance(snapshot, dict):
                probs = {"斷刀風險": 0.03, "重刀": 0.12, "正常": 0.70, "輕刀": 0.15}
            else:
                status = str(snapshot.get("Status", "OK")).upper()
                wear = to_float(snapshot.get("Tool_Life", 150), 150)
                if status == "WARN" or wear < 120:
                    probs = {"斷刀風險": 0.70, "重刀": 0.18, "正常": 0.08, "輕刀": 0.04}
                else:
                    probs = {"斷刀風險": 0.03, "重刀": 0.12, "正常": 0.70, "輕刀": 0.15}
            row.append(probs[category])
        matrix.append(row)

    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for row_index, category in enumerate(categories, start=1):
        row_color = palette[category]
        normalized = matrix[row_index - 1]
        max_value = max(normalized) if normalized else 1
        scale = [max(0.08, value / max_value) for value in normalized]
        figure.add_trace(
            go.Heatmap(
                z=[scale],
                x=list(range(len(labels))),
                y=[category],
                showscale=False,
                colorscale=[
                    [0.0, "#ffffff"],
                    [1.0, row_color],
                ],
                zmin=0,
                zmax=1,
                hovertemplate=f"{category}<br>%{{x}}: %{{z:.2f}}<extra></extra>",
            ),
            row=row_index,
            col=1,
        )

    if now_pos is not None and 0 <= now_pos < len(labels):
        figure.add_vline(x=now_pos, line_color="#2ecc71", line_width=3, line_dash="solid")
        figure.add_annotation(x=now_pos, y=1.02, xref="x", yref="paper", text="NOW", showarrow=False, font=dict(color="#2ecc71", size=12))
        if now_pos > 0:
            figure.add_annotation(x=0, y=1.02, xref="x", yref="paper", text="已加工", showarrow=False, font=dict(color="#1f77b4", size=12))
        if now_pos + 1 < len(labels):
            figure.add_annotation(x=len(labels) - 1, y=1.02, xref="x", yref="paper", text="推測未來", showarrow=False, font=dict(color="#f39c12", size=12))

    # 使用數字 x 軸並顯示對應的標籤文字
    figure.update_xaxes(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels)
    figure.update_layout(title="狀態分布圖", margin=dict(l=16, r=16, t=42, b=16), height=260, autosize=False)
    figure.update_yaxes(showticklabels=True)
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
    figure.update_layout(title="RA 盒狀圖", margin=dict(l=16, r=16, t=42, b=16), height=230, autosize=False, showlegend=False)
    return figure


def build_history_box_figure(history, key, default_value, labels=None, now_pos=None):
    # history: 列表，為 build_csv_window 返回的快照序列（不環回）
    figure = go.Figure()
    # 逐一檢查 history 中的每一個位置；若該位置有明確的 RA_P* 欄位，才顯示該位置的 box
    for idx, snap in enumerate(history):
        if not isinstance(snap, dict):
            continue
        has_explicit = any(f"RA_P{i+1}" in snap for i in range(6))
        if not has_explicit:
            # 若沒有明確推論點，跳過（使用者要求：沒有以前資料就不要顯示）
            continue
        pts = infer_ra_points(snap, point_count=6)
        label = labels[idx] if (labels and idx < len(labels)) else f"Stage {idx}"
        # 若 pts 為空則塞入預設值
        if not pts:
            pts = [default_value]
        # NOW 位置用特殊顏色
        if now_pos is not None and idx < now_pos:
            marker_color = "#1f77b4"
        elif now_pos is not None and idx == now_pos:
            marker_color = "#2ecc71"
        elif now_pos is not None and idx > now_pos:
            marker_color = "#f39c12"
        else:
            marker_color = "#6a8caf"
        # 使用數字 x 座標對齊位置（0..5），避免類別軸與 vline 不對齊
        figure.add_trace(go.Box(x=[idx] * len(pts), y=pts, name=label, marker_color=marker_color))

    if now_pos is not None and labels and 0 <= now_pos < len(labels):
        # vline/annotations 使用數字座標（位置索引），x 軸顯示文字標籤
        figure.add_vline(x=now_pos, line_color="#2ecc71", line_width=3, line_dash="solid")
        figure.add_annotation(x=now_pos, y=1.02, xref="x", yref="paper", text="NOW", showarrow=False, font=dict(color="#2ecc71", size=12))
        if now_pos > 0:
            figure.add_annotation(x=0, y=1.02, xref="x", yref="paper", text="已加工", showarrow=False, font=dict(color="#1f77b4", size=12))
        if now_pos + 1 < len(labels):
            figure.add_annotation(x=len(labels) - 1, y=1.02, xref="x", yref="paper", text="推測未來", showarrow=False, font=dict(color="#f39c12", size=12))
    
    figure.add_hline(y=1.5, line_dash="dash", line_color="#d0021b")
    figure.add_hline(y=0.8, line_dash="dot", line_color="#f5a623")
    figure.update_layout(
        title="RA 盒狀圖（時間窗口）",
        margin=dict(l=16, r=16, t=42, b=16),
        height=230,
        autosize=False,
        showlegend=False,
        yaxis_title="RA",
        xaxis_title="時間位置",
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
    figure.update_layout(title="RA 推論品質直方圖", margin=dict(l=16, r=16, t=42, b=16), height=230, autosize=False, showlegend=False, xaxis_title="推論品質", yaxis_title="數量")
    return figure


def build_history_ra_histogram_figure(snapshot, base_value):
    # 顯示當前（NOW）快照的 6 個點位推論品質
    if isinstance(snapshot, dict):
        points = infer_ra_points(snapshot, point_count=6)
    else:
        points = [base_value] * 6
    labels = [f"P{i + 1}" for i in range(6)]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=labels, y=points, marker_color="#1f77b4"))
    figure.add_hline(y=0.8, line_dash="dot", line_color="#f5a623")
    figure.add_hline(y=1.5, line_dash="dash", line_color="#d0021b")
    figure.update_layout(
        title="RA 推論品質圖（本次完成）",
        margin=dict(l=16, r=16, t=42, b=16),
        height=230,
        autosize=False,
        showlegend=False,
        xaxis_title="N 個點",
        yaxis_title="推論品質",
    )
    return figure


def build_status_gauge_figure(batch_offset):
    probability = max(0.0, min(1.0, 0.45 + batch_offset * 0.02))
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "狀態機率 / 指針"},
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
            title={"text": "狀態機率 / 指針"},
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

MQTT_CLIENT = connect_mqtt()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder=str(ROOT / "assets"))

csv_count = len(CSV_SNAPSHOTS)
csv_max = max(0, csv_count - 1)
csv_marks = {i: str(i) for i in range(csv_count)} if csv_count > 0 else {0: "0"}

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H2("CNC 智慧製造頁面"), md=10),
                dbc.Col(dbc.Button("返回", id="return-btn", color="light", className="w-100"), md=2),
            ],
            align="center",
            className="my-2",
        ),
        dcc.Interval(id="mqtt-refresh", interval=1000, n_intervals=0),
        dcc.Interval(id="playback-tick", interval=1000, n_intervals=0, disabled=True),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("批次拉把", className="section-label"),
                        dcc.Slider(
                            id="batch-slider",
                            min=0,
                            max=20,
                            step=1,
                            value=0,
                            marks={0: "0", 5: "5", 10: "10", 15: "15", 20: "20"},
                        ),
                    ],
                    md=12,
                )
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Button("播放 CSV", id="playback-toggle", color="primary", className="w-100"), md=2),
                dbc.Col(
                    [
                        html.Div("CSV 播放索引", className="section-label"),
                        dcc.Slider(
                            id="csv-index",
                            min=0,
                            max=csv_max,
                            step=1,
                            value=3 if csv_count >= 4 else 0,
                            marks=csv_marks,
                            disabled=(csv_count == 0),
                        ),
                    ],
                    md=10,
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(metric_card("刀具磨耗", "0.25", "mm", "ok", "警戒值", card_id="wear"), md=2),
                dbc.Col(metric_card("Toque", "0.8", "Nm", "warn", "警戒值", card_id="toque"), md=2),
                dbc.Col(metric_card("Bending", "80 / 150", "", "ok", "上下界", card_id="bending"), md=2),
                dbc.Col([dcc.Graph(id="status-gauge", config={"displayModeBar": False, "responsive": False}, style={"height": "170px"})], md=2, className="status-gauge-card"),
                dbc.Col(metric_card("RA", "1.5", "um", "warn", "警戒值", card_id="ra"), md=4),
            ],
            className="g-2 mb-3",
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="wear-figure", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                        dcc.Graph(id="toque-figure", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                        dcc.Graph(id="bending-figure", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                    ],
                    className="left-panel",
                ),
                html.Div(
                    [
                        html.Div("STL / 工件示意", className="section-label"),
                        dcc.Loading(children=[html.Div(id="stl-view-area", className="drawing-card")], type="default"),
                        html.Div(id="tool-caption", className="tool-caption"),
                    ],
                    className="center-panel",
                ),
                html.Div(
                    [
                        dcc.Graph(id="status-heatmap", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                        dcc.Graph(id="ra-box", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
                        dcc.Graph(id="ra-trend", config={"displayModeBar": False, "responsive": False}, style={"height": "240px"}),
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
    Output("playback-tick", "disabled"),
    Output("playback-toggle", "children"),
    Input("playback-toggle", "n_clicks"),
)
def toggle_playback(n_clicks):
    if not CSV_SNAPSHOTS:
        return True, "無 CSV 可播放"
    running = bool(n_clicks and n_clicks % 2 == 1)
    return (not running), ("暫停 CSV" if running else "播放 CSV")


@app.callback(
    Output("csv-index", "value"),
    Input("playback-tick", "n_intervals"),
    State("csv-index", "value"),
)
def advance_csv_index(_n_intervals, current_idx):
    if not CSV_SNAPSHOTS:
        return 0
    idx = to_int(current_idx, 0)
    return (idx + 1) % len(CSV_SNAPSHOTS)


@app.callback(
    Output("tool-caption", "children"),
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
    Input("mqtt-refresh", "n_intervals"),
    Input("batch-slider", "value"),
    Input("csv-index", "value"),
)
def update_figures(_n_intervals, batch_slider, csv_index):
    if CSV_SNAPSHOTS:
        current_idx = to_int(csv_index, 0) % len(CSV_SNAPSHOTS)
        history = build_csv_window(CSV_SNAPSHOTS, current_idx)
        window_indices = compute_window_indices(CSV_SNAPSHOTS, current_idx)
        window_labels = compute_window_labels(CSV_SNAPSHOTS, current_idx)
        start = window_indices[0] if window_indices else current_idx
        now_pos = current_idx - start if window_indices else None
        snapshot = history[now_pos] if (now_pos is not None and 0 <= now_pos < len(history) and isinstance(history[now_pos], dict)) else {}
    else:
        snapshot = current_data()
        history = current_history()
        window_indices = list(range(len(history)))
        window_labels = None
        now_pos = None
    wear_base = to_float(snapshot.get("Tool_Life", 150), 150)
    toque_base = to_float(snapshot.get("Toque", snapshot.get("Radius_Comp", 0.8)), 0.8)
    bending_base = to_float(snapshot.get("Bending", snapshot.get("Length_Comp", 80)), 80)
    ra_base = max(0.1, to_float(snapshot.get("RA", snapshot.get("Offset_Z", 1.5)), 1.5))
    tool_code = snapshot.get("Tool_Code", "-")
    status_text = str(snapshot.get("Status", "OK")).upper()
    batch_offset = to_int(batch_slider, 0)

    wear_values = [to_float(item.get("Tool_Life", wear_base), wear_base) for item in history if isinstance(item, dict)]
    toque_values = [to_float(item.get("Toque", toque_base), toque_base) for item in history if isinstance(item, dict)]
    bending_values = [to_float(item.get("Bending", bending_base), bending_base) for item in history if isinstance(item, dict)]
    x_positions = list(range(len(history)))
    wear_figure = build_history_trend_figure("刀具磨耗趨勢", wear_values, 0.25, 0.35, labels=window_labels, now_pos=now_pos, x_values=x_positions)
    toque_figure = build_history_trend_figure("Toque 趨勢", toque_values, 0.8, 1.0, labels=window_labels, now_pos=now_pos, x_values=x_positions)
    bending_figure = build_history_trend_figure("Bending 趨勢", bending_values, 80, 150, labels=window_labels, now_pos=now_pos, x_values=x_positions)
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
    ra_badge = "OK" if ra_base <= 1.5 else "WARN"
    raw_json = html.Pre(json.dumps(snapshot, indent=2, ensure_ascii=False), className="raw-json d-none")
    return (
        f"T-code: T{tool_code}",
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
    )


_stl_component_cache = None


@app.callback(Output("stl-view-area", "children"), Input("mqtt-refresh", "n_intervals"))
def load_stl(_n_intervals):
    global _stl_component_cache
    if _stl_component_cache is not None:
        return _stl_component_cache

    stl_path = ROOT / "!Back Plate.stl"
    try:
        import trimesh

        mesh = trimesh.load_mesh(str(stl_path))
        if mesh.is_empty:
            raise ValueError("empty mesh")

        vertices = mesh.vertices
        faces = mesh.faces
        x, y, z = vertices.T
        i, j, k = faces.T
        figure = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.55, color="#6a8caf")])
        figure.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
        _stl_component_cache = dcc.Graph(figure=figure, config={"displayModeBar": False})
        return _stl_component_cache
    except Exception:
        _stl_component_cache = html.Div(
            [
                html.Div("STL 檢視需安裝 `trimesh` 並放置於 CNC\!Back Plate.stl。"),
                html.Div(str(stl_path), style={"fontSize": "0.8rem", "color": "#666"}),
            ]
        )
        return _stl_component_cache


if __name__ == "__main__":
    app.run(debug=True, port=8050)