import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import trimesh
import json
import os

# ─── 0. 輔助函數 ──────────────────────────────────────────────────────
def get_available_json_files(search_dir=None):
    """在指定目錄及子目錄中列出所有 JSON 檔案"""
    if search_dir is None:
        search_dir = os.path.dirname(__file__)
   
    json_files = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.json'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, search_dir)
                json_files.append({
                    'label': relative_path.replace(os.sep, ' / '),
                    'value': full_path
                })
   
    json_files.sort(key=lambda x: x['label'])
    return json_files

# ─── 1. 資料讀取與管理 ────────────────────────────────────────────────
class DataManager:
    """負責讀取與管理 JSON 資料的類別"""
    def __init__(self, filepath=None):
        if filepath is None:
            self.filepath = None
        elif not os.path.isabs(filepath):
            self.filepath = os.path.join(os.path.dirname(__file__), filepath)
        else:
            self.filepath = filepath
       
        self.data = self._load_data()

    def _load_data(self):
        if self.filepath is None or not os.path.exists(self.filepath):
            return {}
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON 載入錯誤: {e}")
            return {}

    @property
    def geometry(self):
        if "tool" in self.data:
            t = self.data.get("tool", {})
            return {
                "d_cut": float(t.get("diameter", 10.0)),
                "d_shank": float(t.get("diameter", 10.0)),
                "len_cut": float(t.get("cutting_length", 30.0)),
                "len_total": float(t.get("overall_length", 75.0)),
                "n_flutes": int(t.get("flutes", 4)),
                "type": "Flat",
                "r_val": 0.0,
                "brands": t.get("brand", "N/A"),
                "materials": t.get("material", "N/A"),
                "coated": t.get("coated", "N/A"),
                "cut_depth_sim": 5.0,
            }
        return self.data.get("tool_parameter", {})

    @property
    def wear_status(self):
        if "tool" in self.data:
            limit_mm = 0.3
            max_vb = 0.0
            last_cycle = 0
            for cond_key, cond in self.data.get("conditions", {}).items():
                for meas in cond.get("measurement_data", []):
                    cycle = meas.get("cycle", 0)
                    if cycle > last_cycle:
                        last_cycle = cycle
                    for flute in meas.get("flute", []):
                        for fnum, fdata in flute.items():
                            vb = float(fdata.get("VB", 0))
                            if vb > max_vb:
                                max_vb = vb
            Actual_mm = round(max_vb, 4)
            predicted_mm = round(min(max_vb * 1.08, limit_mm), 4)
            if Actual_mm >= 0.25:
                status = "CRITICAL"
            elif Actual_mm >= 0.15:
                status = "WARNING"
            else:
                status = "HEALTHY"
            return {
                "status": status,
                "limit_mm": limit_mm,
                "Actual_mm": Actual_mm,
                "predicted_mm": predicted_mm,
                "last_cycle": last_cycle
            }
        if "selections" in self.data and self.data["selections"]:
            return self.data["selections"][0].get("tool_wear", {})
        return {}

    @property
    def current_process(self):
        if "tool" in self.data:
            conds = self.data.get("conditions", {})
            if conds:
                last_cond = list(conds.values())[-1]
                mp = last_cond.get("machining_parameter", {})
                return {
                    "spindle_rpm": mp.get("spindle_speed", 0),
                    "feed_mm_min": mp.get("feed_rate", 0),
                    "ap": mp.get("ap", 0),
                    "ae": mp.get("ae", 0),
                }
            return {}
        if "selections" in self.data and self.data["selections"]:
            sel = self.data["selections"][0]
            if "nc_process" in sel:
                return sel["nc_process"]
            if "nc" in sel and "nc_process" in sel["nc"]:
                return sel["nc"]["nc_process"]
        return {}

    @property
    def decisions(self):
        if "tool" in self.data:
            return []
        if "selections" in self.data and self.data["selections"]:
            return self.data["selections"][0].get("decision_candidates", [])
        return {}

    @property
    def job_info(self):
        if "tool" in self.data:
            return {"time_tag": "N/A", "nc_file": "N/A"}
        return self.data.get("job", {})

    @property
    def machine_info(self):
        if "tool" in self.data:
            return {"native_name": "N/A", "mcid": "N/A", "controller": "N/A"}
        return self.data.get("machine", {})

    @property
    def holder_info(self):
        if "tool" in self.data:
            return {"holder_id": "N/A", "type": "N/A"}
        return self.data.get("tool_holder", {})

def load_prediction_csv(filepath):
    """Load the PGNN result file and validate the columns used by the dashboard."""
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Prediction CSV not found: {filepath}")

    prediction_df = pd.read_csv(filepath)
    required_columns = {'Cycle', 'Actual_Wear'}
    missing_columns = required_columns.difference(prediction_df.columns)
    if missing_columns:
        raise ValueError(f"Prediction CSV is missing columns: {', '.join(sorted(missing_columns))}")
    return prediction_df


# Input files are deliberately configured in the __main__ block below.
dm = DataManager()

# ─── 2. 立銑刀幾何模型生成器 ───────────────────────────────────────────
class EndMillGenerator:
    """生成立銑刀（End Mill）3D網格模型的核心類別"""
    def __init__(self, params):
        self.d_cut = float(params.get('d_cut', 10.0))
        self.d_shank = float(params.get('d_shank', 10.0))
        self.len_cut = float(params.get('len_cut', 30.0))
        self.len_total = float(params.get('len_total', 75.0))
        self.n_flutes = int(params.get('n_flutes', 4))
        self.type = params.get('type', 'Flat')
        self.r_val = float(params.get('r_val', 0.0))
        self.helix_angle = 35.0
        self.flute_depth = 0.4

    def _get_base_radius(self, z):
        r_cut = self.d_cut / 2.0
        r_shank = self.d_shank / 2.0
        z_trans = self.len_total - self.len_cut
        if z < z_trans:
            return r_shank
        dist = self.len_total - z
        if self.type == 'Flat':
            return r_cut
        elif self.type == 'Ball':
            if dist < r_cut:
                return np.sqrt(max(0, r_cut**2 - (r_cut - dist)**2))
        elif self.type == 'Radius':
            if dist < self.r_val:
                return (r_cut - self.r_val) + np.sqrt(max(0, self.r_val**2 - (self.r_val - dist)**2))
        return r_cut

    def generate_mesh(self):
        z_res, theta_res = 100, 120
        z_trans = self.len_total - self.len_cut
        z_arr = np.concatenate([
            np.linspace(0, z_trans, 20, endpoint=False),
            np.linspace(z_trans, self.len_total, 80)
        ])
        t_arr = np.linspace(0, 2 * np.pi, theta_res, endpoint=False)
        Z, T = np.meshgrid(z_arr, t_arr)
        twist = np.tan(np.radians(self.helix_angle)) / (self.d_cut * 0.5)
        R = np.zeros_like(Z)
        Intensity = np.zeros_like(Z)
        rows, cols = Z.shape
        for i in range(rows):
            phi = (t_arr[i] * self.n_flutes) % (2 * np.pi)
            x = phi / (2 * np.pi)
            shape_factor = 1.0 if x > 0.9 else (x / 0.9)**2.5
            r_mod = (1.0 - self.flute_depth) + self.flute_depth * shape_factor
            for j in range(cols):
                z_val = Z[i, j]
                base_r = self._get_base_radius(z_val)
                if z_val >= z_trans:
                    R[i, j] = base_r * r_mod
                    Intensity[i, j] = x
                else:
                    R[i, j] = base_r
                    Intensity[i, j] = 0.0
        X = R * np.cos(T + (Z - z_trans) * twist)
        Y = R * np.sin(T + (Z - z_trans) * twist)
        verts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        intensities = Intensity.flatten()
        faces = []
        for i in range(rows):
            for j in range(cols - 1):
                p1 = i * cols + j
                p2 = i * cols + (j + 1)
                next_i = (i + 1) % rows
                p3 = next_i * cols + j
                p4 = next_i * cols + (j + 1)
                faces.append([p1, p3, p2])
                faces.append([p2, p3, p4])
        top_center_idx = len(verts)
        verts = np.vstack([verts, [0, 0, self.len_total]])
        intensities = np.append(intensities, 1.0)
        for i in range(rows):
            curr = i * cols + (cols - 1)
            next_ = ((i + 1) % rows) * cols + (cols - 1)
            faces.append([top_center_idx, curr, next_])
        bottom_center_idx = len(verts)
        verts = np.vstack([verts, [0, 0, 0]])
        intensities = np.append(intensities, 0.0)
        for i in range(rows):
            curr = i * cols
            next_ = ((i + 1) % rows) * cols
            faces.append([bottom_center_idx, next_, curr])
        return verts, np.array(faces), intensities

    def get_trimesh_object(self):
        verts, faces, intensities = self.generate_mesh()
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        return mesh, verts, faces, intensities

# ─── 3. Dash 應用程式 ──────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Tool Wear Evaluation"
STYLE_CONTAINER = {'display': 'flex', 'height': '100vh', 'fontFamily': 'Segoe UI, Arial', 'fontSize': '18px', 'overflow': 'hidden'}
STYLE_LEFT = {'flex': '0.25','minWidth': 0, 'backgroundColor': '#fff', 'position': 'relative'}
STYLE_RIGHT = {'flex': '1', 'minWidth': '600px', 'backgroundColor': '#f8f9fa', 'padding': '24px', 'overflowY': 'auto', 'borderLeft': '1px solid #ddd'}
STYLE_CARD = {'background': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)', 'marginBottom': '18px'}
STYLE_ROW = {'display': 'flex', 'justifyContent': 'space-between', 'borderBottom': '1px dashed #eee', 'padding': '8px 0', 'fontSize': '17px'}
status_map = {
    'HEALTHY': '#28a745',
    'WARNING': '#fd7e14',
    'CRITICAL': '#dc3545',
    'UNKNOWN': '#6c757d'
}
app_store = {
    'current_data_manager': dm,
    'prediction_df': pd.DataFrame(),
    'json_filepath': None,
    'prediction_csv_filepath': None,
}


def configure_dashboard(json_filepath, prediction_csv_filepath):
    """Set the dashboard input files before starting Dash."""
    json_filepath = os.path.abspath(json_filepath)
    prediction_csv_filepath = os.path.abspath(prediction_csv_filepath)
    app_store['current_data_manager'] = DataManager(json_filepath)
    app_store['prediction_df'] = load_prediction_csv(prediction_csv_filepath)
    app_store['json_filepath'] = json_filepath
    app_store['prediction_csv_filepath'] = prediction_csv_filepath
    # The layout is already defined when this function is called from __main__.
    app.layout.children[0].data = {'filepath': json_filepath}


def get_live_wear_status(data_manager, prediction_df):
    """Return status values, using the newest actual and forecast CSV values."""
    wear_status = data_manager.wear_status.copy()
    if prediction_df is None or prediction_df.empty:
        return wear_status

    actual_wear = prediction_df.get('Actual_Wear')
    if actual_wear is not None and actual_wear.notna().any():
        wear_status['Actual_mm'] = round(float(actual_wear.dropna().iloc[-1]), 4)

    theoretical_wear = prediction_df.get('Theoretical_Wear')
    if theoretical_wear is not None and theoretical_wear.notna().any():
        wear_status['Theoretical_mm'] = round(float(theoretical_wear.dropna().iloc[-1]), 4)

    for column in ('Final_Pred_c+2', 'Final_Pred_c+1', 'Final_Pred_c'):
        prediction = prediction_df.get(column)
        if prediction is not None and prediction.notna().any():
            wear_status['predicted_mm'] = round(float(prediction.dropna().iloc[-1]), 4)
            break

    Actual_mm = wear_status.get('Actual_mm', 0)
    wear_status['status'] = (
        'CRITICAL' if Actual_mm >= 0.25 else
        'WARNING' if Actual_mm >= 0.15 else 'HEALTHY'
    )
    return wear_status


def get_wear_values_at_cycle(prediction_df, cycle, fallback_status):
    """Get the latest measurement and prediction available at a machining cycle."""
    Actual_mm = fallback_status.get('Actual_mm', 0.0)
    Theoretical_mm = fallback_status.get('Theoretical_mm', 0.0)
    predicted_mm = fallback_status.get('predicted_mm', 0.0)
    if prediction_df is None or prediction_df.empty:
        return Actual_mm, Theoretical_mm, predicted_mm

    available_data = prediction_df[prediction_df['Cycle'] <= cycle]
    if available_data.empty:
        return Actual_mm, Theoretical_mm, predicted_mm

    actual_wear = available_data.get('Actual_Wear')
    if actual_wear is not None and actual_wear.notna().any():
        Actual_mm = float(actual_wear.dropna().iloc[-1])

    theoretical_wear = available_data.get('Theoretical_Wear')
    if theoretical_wear is not None and theoretical_wear.notna().any():
        Theoretical_mm = float(theoretical_wear.dropna().iloc[-1])

    for column in ('Final_Pred_c+2', 'Final_Pred_c+1', 'Final_Pred_c'):
        prediction = available_data.get(column)
        if prediction is not None and prediction.notna().any():
            predicted_mm = float(prediction.dropna().iloc[-1])
            break
    return Actual_mm, Theoretical_mm, predicted_mm


def estimate_tool_rul(prediction_df, max_wear=0.3, minutes_per_cycle=2):
    """Estimate tool RUL from the furthest PGNN forecast available in the CSV."""
    empty_result = {
        'current_cycle': 0, 'additional_cycles': 0, 'failure_cycle': 0,
        'rul_minutes': 0, 'change_tool': False,
    }
    if prediction_df is None or prediction_df.empty:
        return empty_result

    current_cycle = int(prediction_df['Cycle'].max())
    actual_wear = prediction_df.get('Actual_Wear')
    actual_limit_reached = actual_wear is not None and (actual_wear >= max_wear).any()
    prediction_series = None
    for column in ('Final_Pred_c+2', 'Final_Pred_c+1', 'Final_Pred_c'):
        if column in prediction_df and prediction_df[column].notna().sum() >= 2:
            prediction_series = prediction_df[['Cycle', column]].dropna().sort_values('Cycle')
            break
    if prediction_series is None:
        return {**empty_result, 'current_cycle': current_cycle}

    cycles = prediction_series['Cycle'].to_numpy(dtype=float)
    wear = prediction_series.iloc[:, 1].to_numpy(dtype=float)
    reached_limit = wear >= max_wear
    if reached_limit.any():
        failure_cycle = int(cycles[np.flatnonzero(reached_limit)[0]])
    else:
        # Use the most recent two forecasts, matching tool_RUL_estimation.py.
        slope = (wear[-1] - wear[-2]) / (cycles[-1] - cycles[-2])
        if slope <= 0:
            return {**empty_result, 'current_cycle': current_cycle}
        failure_cycle = int(np.ceil(cycles[-1] + (max_wear - wear[-1]) / slope))

    additional_cycles = max(0, failure_cycle - current_cycle)
    return {
        'current_cycle': current_cycle,
        'additional_cycles': additional_cycles,
        'failure_cycle': failure_cycle,
        'rul_minutes': additional_cycles * minutes_per_cycle,
        'change_tool': actual_limit_reached or wear[-1] >= max_wear,
    }

# ─── Layout ────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id='file-store', data={'filepath': app_store['json_filepath']}),
    html.Div([
        html.Div(id='3d-container', children=[dcc.Graph(id='3d-preview', style={'height': '100vh'})], style={'display': 'none', 'height': '100vh'}),
        # html.Div([
        #     html.Div("Input files are configured in quality_summary.py (__main__).",
        #              style={'fontSize': '15px', 'color': '#555'})
        # ], style={'position': 'absolute', 'top': '20px', 'left': '20px', 'background': 'rgba(255,255,255,0.95)', 'padding': '8px 12px', 'borderRadius': '5px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'})
    ], style=STYLE_LEFT),
    html.Div([
        html.Details([html.Summary("Machining Parameters", style={'fontWeight': 'bold', 'fontSize': '28px', 'cursor': 'pointer', 'outline': 'none'}), html.Div(id='machining-params-container', children=[])], style=STYLE_CARD, open=False),
        html.Details([html.Summary("Live Status", style={'fontWeight': 'bold', 'fontSize': '28px', 'cursor': 'pointer', 'outline': 'none'}), html.Div(id='status-trends-container', children=[])], style={**STYLE_CARD, 'minHeight': '680px'}, open=True),
        html.Details([
            html.Summary([
                html.Span("Tool RUL"),
                html.Span(id='change-tool-label'),
            ], style={'fontWeight': 'bold', 'fontSize': '28px', 'cursor': 'pointer', 'outline': 'none', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
            html.Div(id='tool-rul-container', children=[]),
        ], style=STYLE_CARD, open=True),
        html.Details([html.Summary("Export", style={'fontWeight': 'bold', 'fontSize': '28px', 'cursor': 'pointer', 'outline': 'none'}), html.Div([
            html.H4("Time Prediction", style={'marginTop': '10px', 'marginBottom': '10px'}),
            dcc.Slider(id='time-slider', min=0, max=100, value=0, marks={0: 'Now', 100: 'End'}, tooltip={"placement": "bottom", "always_visible": True}),
            html.Hr(style={'margin': '15px 0'}),
            html.Button("Download STL Model", id="btn_download", style={'width': '100%', 'padding': '12px', 'backgroundColor': '#6c757d', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '17px'}),
            dcc.Download(id="download-stl")
        ], style={'marginTop': '10px'})], style=STYLE_CARD, open=False),
    ], style=STYLE_RIGHT)
], style=STYLE_CONTAINER)

# ─── 4. Callbacks ──────────────────────────────────────────────────────
@app.callback(Output('3d-container', 'style'), Input('file-store', 'data'))
def toggle_3d_visibility(file_store_data):
    if file_store_data and file_store_data.get('filepath'):
        return {'display': 'block', 'height': '100vh'}
    return {'display': 'none', 'height': '100vh'}

@app.callback(
    Output('machining-params-container', 'children'),
    Output('status-trends-container', 'children'),
    Output('tool-rul-container', 'children'),
    Output('change-tool-label', 'children'),
    Output('change-tool-label', 'style'),
    Input('file-store', 'data'),
    prevent_initial_call=False
)
def update_dashboard(file_store_data):
    if not file_store_data or not file_store_data.get('filepath'):
        empty_layout = html.Div("Configure the JSON and prediction CSV paths in __main__.", style={'textAlign': 'center', 'padding': '20px', 'color': '#999', 'fontSize': '14px'})
        return empty_layout, empty_layout, empty_layout, '', {'display': 'none'}

    try:
        new_dm = app_store['current_data_manager']

        is_new_format = "tool" in new_dm.data
        label_style = {'fontWeight': '600', 'color': '#555', 'minWidth': '160px', 'display': 'inline-block', 'fontSize': '24px'}
        value_style = {'fontFamily': 'Consolas, monospace', 'color': '#222'}

        if is_new_format:
            tool = new_dm.data.get("tool", {})
            conds = new_dm.data.get("conditions", {})
            last_key = list(conds.keys())[-1] if conds else "N/A"
            last_mp = conds.get(last_key, {}).get("machining_parameter", {}) if last_key != "N/A" else {}
            left_column = html.Div([
                html.Div([html.Span("Tool", style=label_style), html.Span(f"{tool.get('brand', 'N/A') }", style=value_style)], style=STYLE_ROW),
                html.Div([html.Span("Material", style=label_style), html.Span(f"{tool.get('material', 'N/A')}", style=value_style)], style=STYLE_ROW),
                html.Div([html.Span("Coated", style=label_style), html.Span(f"{tool.get('coated', 'N/A')}", style=value_style)], style=STYLE_ROW),

                html.Div([html.Span("Diameter", style=label_style), html.Span(f"{tool.get('diameter', 0)} mm", style=value_style)], style=STYLE_ROW),
                # html.Div([html.Span("Cutting / Total Length", style=label_style), html.Span(f"{tool.get('cutting_length', 0)} / {tool.get('overall_length', 0)} mm", style=value_style)], style=STYLE_ROW),
                # html.Div([html.Span("Last Condition", style=label_style), html.Span(str(last_key), style=value_style)], style=STYLE_ROW),
                # html.Div([html.Span("ap × ae (mm)", style=label_style), html.Span(f"{last_mp.get('ap', 0)} × {last_mp.get('ae', 0)}", style=value_style)], style=STYLE_ROW),
                # html.Div([html.Span("Spindle / Feed", style=label_style), html.Span(f"{last_mp.get('spindle_speed', 0)} rpm / {last_mp.get('feed_rate', 0)} mm/min", style=value_style)], style=STYLE_ROW),
            ], style={'flex': '1', 'paddingRight': '15px'})
        else:
            left_column = html.Div([
                html.Div([html.Span("Job Time Tag", style=label_style), html.Span(new_dm.job_info.get('time_tag', 'N/A'), style=value_style)], style=STYLE_ROW),
                html.Div([html.Span("NC Program", style=label_style), html.Span(new_dm.job_info.get('nc_file', 'N/A'), style=value_style)], style=STYLE_ROW),
                html.Div([html.Span("Machine", style=label_style), html.Span(f"{new_dm.machine_info.get('native_name', 'N/A')} ({new_dm.machine_info.get('mcid', 'N/A')})", style=value_style)], style=STYLE_ROW),
                html.Div([html.Span("Controller", style=label_style), html.Span(new_dm.machine_info.get('controller', 'N/A'), style=value_style)], style=STYLE_ROW),
                html.Div([html.Span("Tool Holder", style=label_style), html.Span(f"{new_dm.holder_info.get('holder_id', 'N/A')} ({new_dm.holder_info.get('type', 'N/A')})", style=value_style)], style=STYLE_ROW),
            ], style={'flex': '1', 'paddingRight': '15px'})

        right_column_children = []
        if new_dm.geometry and len(new_dm.geometry) > 0:
            geometry_items = [
                # ("Tool", f"{new_dm.geometry.get('brands', 'N/A')} {new_dm.geometry.get('materials', 'N/A')} {new_dm.geometry.get('type', 'N/A')}"),
                # ("Cutting Diameter", f"{new_dm.geometry.get('d_cut', 0):.2f} mm"),
                # ("Shank Diameter", f"{new_dm.geometry.get('d_shank', 0):.2f} mm"),
                ("Length of Cut", f"{tool.get('cutting_length', 0)} mm"),
                ("Total Length", f"{tool.get('overall_length', 0)} mm"),
                ("Flutes", str(tool.get('flutes', 'N/A'))),
                ("Rake Angle / Helix Angle", f"{tool.get('rake_angle', 'N/A')}° / {tool.get('helix_angle', 'N/A')}°"),
                # ("Corner Radius (R)", f"{new_dm.geometry.get('r_val', 0):.2f} mm"),
            ]
            for label, value in geometry_items:
                right_column_children.append(html.Div([html.Span(label, style=label_style), html.Span(value, style=value_style)], style=STYLE_ROW))
        right_column = html.Div(right_column_children, style={'flex': '1', 'paddingLeft': '15px'})

        machining_content = html.Div([left_column, right_column], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '30px', 'marginTop': '16px', 'padding': '0 10px'})

        wear_status = get_live_wear_status(new_dm, app_store.get('prediction_df'))
        current_status = wear_status.get('status', 'UNKNOWN')
        status_color = status_map.get(current_status, '#6c757d')
        rul = estimate_tool_rul(
            app_store.get('prediction_df'),
            max_wear=float(wear_status.get('limit_mm', 0.3)),
        )
        rul_style = {
            'marginTop': '14px', 'padding': '16px', 'borderRadius': '8px',
            'border': '2px solid #dc3545' if rul['change_tool'] else '1px solid #cbd5e1',
            'backgroundColor': '#fff1f2' if rul['change_tool'] else '#f8fafc',
        }
        rul_content = html.Div([
            html.Div([
                # html.Div([html.Span("⚠ Change Tool ", style={'color': "#FF0000",'fontSize': '23px', 'fontWeight': 'bold'})]),
                html.Div([html.Span("Current: ", style={'color': '#000000','fontSize': '28px', 'fontWeight': 'bold'}), html.Strong(f"{rul['current_cycle']} cycles", style={'fontSize': '28px'})]),
                html.Div([html.Span("Additional: ", style={'color': '#000000','fontSize': '28px', 'fontWeight': 'bold'}), html.Strong(f"{rul['additional_cycles']} cycles", style={'fontSize': '28px'})]),
                html.Div([html.Span("Predicted Failure: ", style={'color': '#000000','fontSize': '28px', 'fontWeight': 'bold'}), html.Strong(f"{rul['failure_cycle']} cycles", style={'fontSize': '28px'})]),
                html.Div([html.Span("RUL: ", style={'color': '#000000','fontSize': '28px', 'fontWeight': 'bold'}), html.Strong(f"{rul['rul_minutes']} min", style={'fontSize': '28px'})]),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '16px', 'fontSize': '24px'}),
        ], style=rul_style)

        status_trends_content = [
            html.Div([
                html.Div([html.Span("Limit: ", style={'fontSize': '28px', 'fontWeight': 'bold'}), html.Strong(f"{wear_status.get('limit_mm', 0.3)} mm", style={'fontSize': '28px', 'color': "#ff0000"})]),
                html.Div([html.Span("Actual: ", style={'fontSize': '28px', 'fontWeight': 'bold'}), html.Strong(f"{wear_status.get('Actual_mm', 0)} mm", id='Actual-value', style={'fontSize': '28px'})]),
                html.Div([html.Span("Theoretical: ", style={'fontSize': '28px', 'fontWeight': 'bold'}), html.Strong(f"{wear_status.get('Theoretical_mm', 0)} mm", id='Theoretical-value', style={'fontSize': '28px'})]),
                html.Div([html.Span("Predicted: ", style={'fontSize': '28px', 'fontWeight': 'bold'}), html.Strong(f"{wear_status.get('predicted_mm', 0)} mm", id='predicted-value', style={'fontSize': '28px'})]),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '16px', 'fontSize': '17px'}),
            dcc.Graph(id='prognostic-chart', config={'displayModeBar': False}, style={'height': '570px'})
        ]

        change_tool_style = {
            'color': '#dc3545', 'fontSize': '28px', 'fontWeight': '700',
            'marginRight': '12px', 'display': 'inline',
        }
        return (
            machining_content,
            html.Div(status_trends_content, style={'marginTop': '10px'}),
            rul_content,
            '⚠ Change Tool' if rul['change_tool'] else '',
            change_tool_style if rul['change_tool'] else {'display': 'none'},
        )
    except Exception as e:
        print(f"檔案載入錯誤: {e}")
        error_msg = html.Div(f"Error loading file: {str(e)}", style={'textAlign': 'center', 'padding': '20px', 'color': 'red', 'fontSize': '14px'})
        return error_msg, error_msg, error_msg, '', {'display': 'none'}

@app.callback(
    Output('3d-preview', 'figure'),
    Output('prognostic-chart', 'figure'),
    Output('Actual-value', 'children'),
    Output('Theoretical-value', 'children'),
    Output('predicted-value', 'children'),
    Output('download-stl', 'data'),
    Input('time-slider', 'value'),
    Input('btn_download', 'n_clicks'),
    Input('file-store', 'data'),
    prevent_initial_call=False
)
def update_all(time_value, n_clicks, file_store_data):
    if not file_store_data or not file_store_data.get('filepath'):
        empty_fig = go.Figure()
        empty_fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='white'), margin=dict(l=0, r=0, t=0, b=0))
        return empty_fig, go.Figure(), dash.no_update, dash.no_update, dash.no_update, dash.no_update
   
    current_dm = app_store.get('current_data_manager', dm)
    params = current_dm.geometry
    cut_depth = float(params.get('cut_depth_sim', 5.0))
    len_total = float(params.get('len_total', 75.0))
    live_wear_status = get_live_wear_status(current_dm, app_store.get('prediction_df'))
    limit_mm = float(live_wear_status.get("limit_mm", 0.3))
    Actual_mm = float(live_wear_status.get("Actual_mm", 0.0))
    Theoretical_mm = float(live_wear_status.get("Theoretical_mm", 0.0))
    predicted_mm = float(live_wear_status.get("predicted_mm", 0.0))
    trigger_id = ctx.triggered_id
    gen = EndMillGenerator(params)

    if trigger_id == 'btn_download':
        mesh_obj, _, _, _ = gen.get_trimesh_object()
        stl_bytes = mesh_obj.export(file_type='stl')
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dcc.send_bytes(stl_bytes, "wear_simulation.stl")
    
    verts, faces, intensity = gen.generate_mesh()
    radii = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    r_cut = params.get('d_cut', 10.0) / 2.0
    is_flank_area = radii > (r_cut * 0.6)
    t_factor = time_value / 100.0
    simulated_wear_mm = predicted_mm + (t_factor * (limit_mm * 1.1 - predicted_mm))
    wear_ratio = simulated_wear_mm / limit_mm
    z_wear_limit = len_total - cut_depth
    in_cut_zone = verts[:, 2] >= z_wear_limit
    base_threshold = 1.001
    current_threshold = base_threshold - (wear_ratio * 0.05)
    colors = intensity * 0.6
    is_wear = (intensity > current_threshold) & in_cut_zone & is_flank_area
    if np.any(is_wear):
        wear_intensity = 0.6 + (wear_ratio * 0.4)
        colors[is_wear] = np.clip(wear_intensity, 0.6, 1.0)
    custom_colorscale = [
        [0.00, 'rgb(30,  60,  140)'], [0.40, 'rgb(50,  100, 200)'], [0.60, 'rgb(100, 200, 255)'],
        [0.70, 'rgb(255, 255, 120)'], [0.80, 'rgb(255, 180,  60)'], [1.00, 'rgb(220,  40,  40)']
    ]
    fig_3d = go.Figure(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], intensity=colors, colorscale=custom_colorscale, cmin=0.0, cmax=1.0, showscale=False, flatshading=True, lighting=dict(ambient=0.6, roughness=0.1, specular=0.5)))
    fig_3d.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', bgcolor='white', camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1, y=0, z=len_total / 50 + 1.5), center=dict(x=0, y=0, z=1.2))), margin=dict(l=0, r=0, t=0, b=0))
    
    pred_df = app_store.get('prediction_df')
    fig_chart = go.Figure()
    displayed_Actual = Actual_mm
    displayed_Theoretical = Theoretical_mm
    displayed_predicted = predicted_mm


    if pred_df is not None and not pred_df.empty:
        df = pred_df.copy().dropna(subset=['Cycle'])
        max_cycle = int(df['Cycle'].max())
        current_cycle = max(1, int(round((time_value / 100.0) * max_cycle)))
        displayed_Actual, displayed_Theoretical, displayed_predicted = get_wear_values_at_cycle(
            df, current_cycle, live_wear_status
        )
        fig_chart.add_trace(go.Scatter(x=df[df['Actual_Wear'].notna()]['Cycle'], y=df[df['Actual_Wear'].notna()]['Actual_Wear'], mode='markers+lines', name='Actual', line=dict(color="#0804f8", width=2.5, dash='dash'), marker=dict(symbol='square', size=8, color='#1f77b4')))
        fig_chart.add_trace(go.Scatter(x=df[df['Theoretical_Wear'].notna()]['Cycle'], y=df[df['Theoretical_Wear'].notna()]['Theoretical_Wear'], mode='markers+lines', name='Theoretical', line=dict(color='#2ca02c', width=2.5), marker=dict(symbol='circle', size=7, color='#2ca02c')))
        fig_chart.add_trace(go.Scatter(x=df[df['Final_Pred_c'].notna()]['Cycle'], y=df[df['Final_Pred_c'].notna()]['Final_Pred_c'], mode='markers+lines', name='Predicted (C)', line=dict(color='#9467bd', width=2, dash='dash'), marker=dict(symbol='x', size=9, color='#9467bd')))
        fig_chart.add_trace(go.Scatter(x=df[df['Final_Pred_c+1'].notna()]['Cycle'], y=df[df['Final_Pred_c+1'].notna()]['Final_Pred_c+1'], mode='markers+lines', name='Predicted (C+1)', line=dict(color='#ff7f0e', width=2, dash='dash'), marker=dict(symbol='triangle-up', size=9, color='#ff7f0e')))
        fig_chart.add_trace(go.Scatter(x=df[df['Final_Pred_c+2'].notna()]['Cycle'], y=df[df['Final_Pred_c+2'].notna()]['Final_Pred_c+2'], mode='markers+lines', name='Predicted (C+2)', line=dict(color='#d62728', width=2, dash='dash'), marker=dict(symbol='triangle-down', size=9, color='#d62728')))
        # Match the condition-background presentation used by the PGNN plots.
        condition_colors = ['#dbeafe', '#fee2e2', '#d1fae5', '#fef3c7', '#f3e8ff']
        conditions = current_dm.data.get('conditions', {})
        for index, (condition_name, condition) in enumerate(conditions.items()):
            cycles = [item.get('cycle') for item in condition.get('measurement_data', [])
                      if item.get('cycle') is not None]
            if cycles:
                fig_chart.add_vrect(
                    x0=min(cycles) - 0.5, x1=max(cycles) + 0.5,
                    fillcolor=condition_colors[index % len(condition_colors)], opacity=0.35,
                    layer='below', line_width=0,
                    annotation_text=f'Condition {condition_name}',
                    annotation_position='top left', annotation_font_size=20,
                    annotation_font_color='black'
                )
        sim_cycle = int((time_value / 100.0) * max_cycle)
        fig_chart.add_vline(x=sim_cycle, line_width=2.5, line_dash="dot", line_color="#555555", annotation_text=f"", annotation_position="top right")
        y_values = df[['Actual_Wear', 'Theoretical_Wear', 'Final_Pred_c', 'Final_Pred_c+1', 'Final_Pred_c+2']].to_numpy(dtype=float)
        y_max = max(limit_mm, np.nanmax(y_values)) * 1.1
        fig_chart.update_layout(title=dict(text="PGNN Tool Wear Prediction (NKUST-B)", font=dict(size=24, color='#333')), xaxis_title="Machining Cycle", yaxis_title="Tool Wear VB (mm)", font=dict(size=17), margin=dict(l=65, r=35, t=70, b=60), height=570, legend=dict(orientation="h", y=-0.2, x=0.02, font=dict(size=15)), template="plotly_white", xaxis=dict(range=[0, max_cycle + 1], dtick=2, showgrid=True, gridcolor='#e0e0e0'), yaxis=dict(range=[0, y_max], showgrid=True, gridcolor='#e0e0e0'), hovermode="x unified")
    else:
        time_horizon = 60
        fig_chart.add_trace(go.Scatter(x=[0, time_horizon], y=[limit_mm, limit_mm], mode='lines', name='Limit', line=dict(color='red', width=2, dash='dash')))
        fig_chart.add_trace(go.Scatter(x=[0], y=[Actual_mm], mode='markers', name='Actual', marker=dict(color='black', size=8, symbol='circle')))
        fig_chart.add_trace(go.Scatter(x=[0], y=[Theoretical_mm], mode='markers', name='Theoretical', marker=dict(color='blue', size=10, symbol='square')))
        fig_chart.add_trace(go.Scatter(x=[0], y=[predicted_mm], mode='markers', name='Predicted Start', marker=dict(color='black', size=10, symbol='diamond')))
        current_sim_time = (time_value / 100.0) * time_horizon
        fig_chart.add_vline(x=current_sim_time, line_width=2, line_dash="dot", line_color="#555")
        fig_chart.update_layout(xaxis_title="Future Time (min)", yaxis_title="Wear (mm)", font=dict(size=17), margin=dict(l=65, r=35, t=60, b=60), height=570, legend=dict(orientation="h", y=-0.2, font=dict(size=15)), template="plotly_white")

    return (
        fig_3d,
        fig_chart,
        f"{displayed_Actual:.4f} mm",
        f"{displayed_Theoretical:.4f} mm",
        f"{displayed_predicted:.4f} mm",
        dash.no_update,
    )

if __name__ == '__main__':
    # Change these two paths when switching to another tool-wear experiment.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    JSON_FILE = os.path.join(BASE_DIR, 'NKUST-C', 'wear_data_C.json')
    PREDICTION_CSV_FILE = os.path.join(
        BASE_DIR, 'case-3', 'result', 'prediction_NKUST-C.csv'
    )
    configure_dashboard(JSON_FILE, PREDICTION_CSV_FILE)
    app.run(debug=True, dev_tools_ui=False)
