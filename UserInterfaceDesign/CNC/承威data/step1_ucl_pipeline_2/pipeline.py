import os
from pathlib import Path
import pandas as pd
import yaml

# 匯入 Step 1 主程式 (負責資料對齊與 UCL 計算，產出 CSV)
from step1_ucl_pipeline_2 import run_pipeline as run_step1_pipeline

# 匯入 Step 2 主程式 (負責繪製 3D 軌跡，產出 HTML)
# 假設你的 step2 檔案名為 step2_render_path_on_stl_autoalign.py
from step2_render_path_on_stl_autoalign import main as render_stl_html

BASE_DIR = Path("./data/exp") 
# 指定你的 STL 模型實際存放位置
STL_MODEL_PATH = Path("./data/325BTM.STL")

def process_experiment_data():
    for exp_num in range(29, 35):
        for sub_num in range(1, 3):
            folder_name = f"EXP-{exp_num}-{sub_num}"
            folder_path = BASE_DIR / folder_name

            if not folder_path.is_dir():
                continue

            print(f"\n--- 開始處理 {folder_name} ---")

            # ==========================================
            # 階段一：執行 Step 1 產生所需的 CSV
            # ==========================================
            try:
                # Know-How: 在資料夾內尋找原始輸入檔。
                # 這裡假設你的 pdata 原始檔特徵是 _trimmed.csv，請依據實際狀況修改
                cnc_files = list((folder_path / "CTL").glob("*_trim_pause.csv"))
                if not cnc_files:
                    print(f"錯誤: 找不到 CNC 原始檔 (*_trim_pause.csv)。跳過。")
                    continue
                cnc_path = cnc_files[0]

                # 定義 Step 1 產出的目標路徑
                reference_path = folder_path / f"{folder_name}_reference.csv"
                aligned = folder_path / f"{folder_name}_aligned.csv"

                # 執行運算
                print("執行 Step 1: 產出 reference 與 aligned...")
                run_step1_pipeline(mode="reference", out_path=reference_path, pdata_path=aligned, cnc_path=cnc_path)

            except Exception as e:
                print(f"執行 Step 1 發生錯誤: {e}")
                continue # 若 CSV 產出失敗，直接跳過此資料夾，不執行 Step 2


            # ==========================================
            # 階段二：執行 Step 2 產生 HTML 視覺化
            # ==========================================
            try:
                print("執行 Step 2: 產出 3D HTML 視覺化...")
                render_stl_html(
                    data_csv=aligned, # 吃 Step 1 剛算出的結果
                    ref_csv=reference_path,       # 吃 Step 1 剛算出的結果
                    stl_path=STL_MODEL_PATH,
                    auto_align_top=True, 
                    eps_xy=0.05,
                    eps_z=0.05,
                    align_candidates="minx_miny_topz,minx_maxy_topz,maxx_miny_topz,maxx_maxy_topz,center_topz",
                    output_dir=folder_path        # 動態指定輸出路徑
                )
            except Exception as e:
                print(f"執行 Step 2 發生錯誤: {e}")
                continue


            # ==========================================
            # 階段三：資料萃取與更新 YAML
            # ==========================================
            try:
                print("執行 Step 3: 更新 YAML 設定檔...")
                # 尋找唯一的 YAML 檔案
                yaml_files = list(folder_path.glob("*.yaml")) + list(folder_path.glob("*.yml"))
                if len(yaml_files) != 1:
                    print(f"錯誤: 預期 1 個 YAML，找到 {len(yaml_files)} 個。跳過 YAML 更新。")
                    continue
                yaml_path = yaml_files[0]

                # 讀取剛剛 Step 1 算出的 aligned_no_ucl_path 進行數值萃取
                df_aligned = pd.read_csv(aligned)
                
                # 若你有額外的 sections.csv，請確保路徑正確並讀取
                sections_csv_path = folder_path / f"{folder_name}_sections.csv"
                if sections_csv_path.exists():
                    df_sections = pd.read_csv(sections_csv_path)
                    sections_value = []
                    if sub_num == 2:
                        sections_value.append(df_sections['Ra_Measured'].iloc[0].item())
                        sections_value.append(df_sections['Ra_Measured'].iloc[-1].item())
                    elif sub_num == 1:
                        sections_value.append(df_sections['Ra_Measured'].iloc[0].item())
                else:
                    sections_value = [] # 預設空值或處理邏輯

                # 讀取 YAML (強制讀取以防舊版 numpy 殘留)
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.unsafe_load(f) or {}

                # 建立或更新 Job 結構
                if 'Job' not in yaml_data:
                    yaml_data['Job'] = {}
                yaml_data['Job']['Quality'] = {
                    'Measurement': sections_value,
                    'Tolerance': 0.8
                }
                
                # 更新數值與 HTML 連結
                html_links = {
                    'Torque_Value_HTML': f'{folder_path}/path_on_stl_torque_value.html',
                    'Torque_UCL_HTML': f'{folder_path}/path_on_stl_torque_ucl.html',
                    'Bending_Value_HTML': f'{folder_path}/path_on_stl_bending_value.html',
                    'Bending_UCL_HTML': f'{folder_path}/path_on_stl_bending_ucl.html'
                }

                if 'ToolRelated' not in yaml_data:
                    yaml_data['ToolRelated'] = {'ToolHolder': {'STH1': {}}}
                if 'Summary' not in yaml_data['ToolRelated']['ToolHolder']['STH1']:
                    yaml_data['ToolRelated']['ToolHolder']['STH1']['Summary'] = {}

                yaml_data['ToolRelated']['ToolHolder']['STH1']['Summary'] = {
                    'MaxTorque': df_aligned['Torque'].max().item(),
                    'rmsTorque': df_aligned['Torque'].mean().item(),
                    'stdTorque': df_aligned['Torque'].std().item(),
                    'MaxBending': df_aligned['BendingX'].max().item(),
                    'VisualizationLinks': html_links  # 寫入相對路徑連結                                      
                }
                
                # 寫回 YAML
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

                print(f"完成: {folder_name} 所有流程")
                
            except Exception as e:
                print(f"執行 Step 3 (YAML 更新) 時發生錯誤: {e}")

if __name__ == "__main__":
    process_experiment_data()