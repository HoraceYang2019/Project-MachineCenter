import os
from pathlib import Path
import pandas as pd
import yaml
from step2_render_path_on_stl_autoalign import main as render_stl_html

# 定義基礎路徑，請替換為你的主資料夾路徑
# 若腳本與 EXP 資料夾在同一層，可使用 Path(".")
BASE_DIR = Path("./data/exp") 
STL_MODEL_PATH = Path("./data/325BTM.STL")
def process_experiment_data():
    # 遍歷 EXP 29 到 34
    for exp_num in range(29, 35):
        # 遍歷子編號 1 到 2
        for sub_num in range(1, 3):
            folder_name = f"EXP-{exp_num}-{sub_num}"
            folder_path = BASE_DIR / folder_name

            # 1. 檢查資料夾是否存在
            if not folder_path.is_dir():
                print(f"找不到資料夾，跳過: {folder_name}")
                continue

            # 2. 尋找唯一的 YAML 檔案
            yaml_files = list(folder_path.glob("*.yaml"))
            # 兼容 .yml 副檔名
            yaml_files.extend(list(folder_path.glob("*.yml")))

            if len(yaml_files) != 1:
                print(f"錯誤: {folder_name} 中找到 {len(yaml_files)} 個 YAML 檔案，預期應只有 1 個。跳過此資料夾。")
                continue
            
            yaml_path = yaml_files[0]

            # 3. 定義並檢查 CSV 檔案路徑
            aligned_csv_name = f"{folder_name}_aligned.csv"
            sections_csv_name = f"{folder_name}_sections.csv"
            
            aligned_csv_path = folder_path / aligned_csv_name
            sections_csv_path = folder_path / sections_csv_name
            
            # 定義 Step 2 需要讀取的 CSV 路徑 (請根據你實際產出的檔名確認)
            aligned_no_ucl_path = folder_path / f"{folder_name}_aligned_no_ucl.csv"
            reference_path = folder_path / f"{folder_name}_reference.csv"
            
            # 觸發繪圖腳本，將輸出目錄指定為當前的 folder_path
            try:
                render_stl_html(
                    data_csv=aligned_no_ucl_path,
                    ref_csv=reference_path,
                    stl_path=STL_MODEL_PATH,
                    auto_align_top=True,  # 使用你程式碼的設定
                    eps_xy=0.05,
                    eps_z=0.05,
                    align_candidates="minx_miny_topz,minx_maxy_topz,maxx_miny_topz,maxx_maxy_topz,center_topz",
                    output_dir=folder_path  # 動態指定輸出路徑
                )
            except Exception as e:
                print(f"渲染 {folder_name} HTML 時發生錯誤: {e}")
                
                
            if not aligned_csv_path.exists() or not sections_csv_path.exists():
                print(f"錯誤: {folder_name} 缺少必要的 CSV 檔案。跳過此資料夾。")
                continue

            try:
                # 4. 讀取 CSV 資料
                df_aligned = pd.read_csv(aligned_csv_path)
                df_sections = pd.read_csv(sections_csv_path)

                # ==========================================
                # 在此處實作你的資料處理邏輯 (Know-How 區塊)
                # 例如：計算平均值、取得特定欄位數值等
                # aligned_value = df_aligned[''].mean()
                sections_value = []
                if sub_num == 1:
                    sections_value.append(df_sections['Ra_Measured'].iloc[0].item())
                    sections_value.append(df_sections['Ra_Measured'].iloc[-1].item())
                elif sub_num == 2 :
                    sections_value.append(df_sections['Ra_Measured'].iloc[0].item())
                # 5. 讀取 YAML 檔案
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.unsafe_load(f) or {}

                html_links = {
                'Torque_Value_HTML': 'path_on_stl_torque_value.html',
                'Torque_UCL_HTML': 'path_on_stl_torque_ucl.html',
                'Bending_Value_HTML': 'path_on_stl_bending_value.html',
                'Bending_UCL_HTML': 'path_on_stl_bending_ucl.html'
                }
                # 在此處更新 YAML 字典內容
                # 直接宣告並賦值給 Quality 字典
                yaml_data['Job']['Quality'] = {
                    # 輸入你的計算結果或從 CSV 取得的數值
                    'Measurement': sections_value,
                    'Tolerance': 0.8
                }
                yaml_data['ToolRelated']['ToolHolder']['STH1']['Summary'] = {
                    # 統一取到數字後三位小數
                    'MaxTorque': round(df_aligned['Torque'].max().item(), 3),
                    'rmsTorque': round(df_aligned['Torque'].mean().item(), 3),
                    'stdTorque': round(df_aligned['Torque'].std().item(), 3),
                    'MaxBending': round(df_aligned['BendingX'].max().item(), 3),
                    'html_links': html_links
                }
                # 6. 將更新後的資料寫回 YAML
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    # default_flow_style=False 確保輸出為標準的分行 YAML 格式，而非 JSON 般的單行格式
                    # allow_unicode=True 確保中文字元不會被轉換為 Unicode 編碼
                    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

                print(f"成功處理並更新: {folder_name}")

            except Exception as e:
                print(f"處理 {folder_name} 時發生錯誤: {e}")

if __name__ == "__main__":
    process_experiment_data()