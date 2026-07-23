import json
import os
import matplotlib.patches as mpatches

os.chdir(os.path.dirname(__file__))

def load_config(config_path: str = "config.json") -> dict:
    """載入 config.json 並自動建立所有必要資料夾"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到設定檔：{config_path}，請先建立 config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    for key in ['pkl_dir', 'model_dir', 'result_dir', 'visualization_dir']:
        dir_path = config['paths'][key]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    return config

def load_wear_data(config: dict, config_key: str = 'data') -> dict:
    """支援 wear_data_condAB.json 的合併結構"""
    wear_json_path = os.path.join(config[config_key]['data_root'], config[config_key]['wear_data_json'])

    if not os.path.exists(wear_json_path):
        print(f"警告：找不到 wear_data.json -> {wear_json_path}")
        return {}
    
    # 檢查檔案名稱是否包含 "QIT"
    filename = os.path.basename(wear_json_path).upper()
    use_vb_fit = "QIT" in filename
    
    with open(wear_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    result = raw_data.copy()
    cycle_indexed_data = {}
    all_measurements = []

    # 支援合併結構：如果有 "conditions" 就從裡面撈資料，否則就當作原本的單一結構
    if "conditions" in raw_data:
        # print("偵測到合併結構")
        for cond_name, cond_data in raw_data["conditions"].items():
            for item in cond_data.get("measurement_data", []):
                item["condition"] = cond_name  # 標記來自哪個條件
                all_measurements.append(item)
    else:
        # 原本的單一結構
        all_measurements = raw_data.get("measurement_data", [])

    # 計算迴圈
    for item in all_measurements:
        cycle = item["cycle"]
        flutes_container = item["flute"][0] if isinstance(item["flute"], list) else item["flute"]
        
        if use_vb_fit:
            # === QIT 模式：直接使用 VB_fit ===
            avg_vb = item.get("VB_fit", 0.0)
        else:
            # === 一般模式：計算四刃平均 ===
            try:
                vb_values = [flutes_container[str(i)]["VB"] for i in range(1, 5)]
                avg_vb = sum(vb_values) / 4.0
            except (KeyError, TypeError, ValueError):
                avg_vb = 0.0
        
        item["VB"] = round(float(avg_vb), 4)
        cycle_indexed_data[cycle] = item

    result["indexed_cycles"] = cycle_indexed_data
    
    return result

def get_cycle_boundaries_by_condition() -> dict:
    """
    根據不同的 condition 分類，找出各自對應的 cycle 起點 (頭) 與終點 (尾)
    
    Returns:
        dict: 結構為 { "condition_name": {"start": int, "end": int}, ... }
    """
    indexed_cycles = CONFIG['wear_data']['indexed_cycles']
    
    # 1. 依據 condition 將所有 cycle 歸類
    # 格式：{ 'condA': [cycle1, cycle2, ...], 'condB': [...] }
    cond_to_cycles = {}
    for cycle_str, info in indexed_cycles.items():
        cond = info.get("condition", "1")
        cycle_val = int(cycle_str)  # 轉成整數以進行正確的數值排序
        
        if cond not in cond_to_cycles:
            cond_to_cycles[cond] = []
        cond_to_cycles[cond].append(cycle_val)
        
    # 2. 找出各 condition 的最小值 (頭) 與最大值 (尾)
    boundaries = {}
    for cond, cycles in cond_to_cycles.items():
        # 確保 cycle 有小到大排序
        sorted_cycles = sorted(cycles)
        boundaries[cond] = {
            "start": sorted_cycles[0],
            "end": sorted_cycles[-1],
            "total_cycles": len(sorted_cycles)  # 順便記錄該條件下共有幾個 cycle
        }
        
    return boundaries

def add_condition_background(ax):
    """
    動態根據 CONFIG 中的 boundaries 設定條件背景色塊 + 在圖表上顯示 Condition 文字
    """
    # 1. 呼叫先前寫好的邊界計算函式
    boundaries = get_cycle_boundaries_by_condition()
    
    # 如果條件數量小於或等於 1 組，直接跳出不畫背景
    if len(boundaries) <= 1:
        return []
    
    # 2. 定義一組好看的粉嫩色系供 Conditions 輪流使用
    color_palette = ['#dbeafe', '#fee2e2', '#d1fae5', '#fef3c7', '#f3e8ff', '#e0f2fe']
    
    legend_patches = []
    seen_labels = set()

    # 先取得目前 y 軸範圍（確保文字位置正確）
    y_max = ax.get_ylim()[1]

    # 3. 依據動態邊界繪製背景
    for i, (cond, range_info) in enumerate(boundaries.items()):
        start = range_info['start']
        end = range_info['end']
        
        # 依序選擇顏色，若 Condition 數量大於顏色庫則循環使用
        color = color_palette[i % len(color_palette)]
        label = f"Condition {cond}"

        # 繪製色塊
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.35, zorder=0)

        # 在色塊中間顯示粗體文字
        mid_point = (start + end) / 2
        ax.text(mid_point, y_max * 0.99, label,
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=16,
                fontweight='bold',
                color='black',
                alpha=0.9)

        if label not in seen_labels:
            legend_patches.append(mpatches.Patch(color=color, alpha=0.35, label=label))
            seen_labels.add(label)
    
    return legend_patches

# ====================== 全局 CONFIG 物件（預設為 None）======================
CONFIG = None

def initialize_config(config_key: str = 'data', reload: bool = True):
    """統一初始化 CONFIG，只需呼叫一次"""
    global CONFIG
    if CONFIG is not None and not reload:
        return CONFIG  # 已經初始化過就直接返回
    
    CONFIG = load_config()
    CONFIG['wear_data'] = load_wear_data(CONFIG, config_key)
    
    
    # 重新包裝字典
    CONFIG['wear_data']['indexed_cycles'] = {
        str(info["cycle"]): {
                "condition": info.get("condition", "1"),  # 預設 condition = "1"
                "VB": info.get("VB")
            }
            for cycle_str, info in CONFIG['wear_data']['indexed_cycles'].items()
        }

    # 建立實際磨耗字典
    CONFIG['actual_wear'] = {
        int(cycle_str): info["VB"] 
        for cycle_str, info in CONFIG['wear_data']['indexed_cycles'].items()
    }
    # print("CONFIG 初始化完成！")
    return CONFIG
    
if __name__ == "__main__":
    initialize_config('data_C')
    # print(CONFIG)
    # print(CONFIG['wear_data']['machining_parameter']['initial_wear'])

    # with open("my_data.json", "w", encoding="utf-8") as f:
    #     json.dump(CONFIG, f, ensure_ascii=False, indent=4)

    boundaries = get_cycle_boundaries_by_condition()
    
    # 3. 漂亮地列印出結果
    print("各加工條件 (Condition) 的 Cycle 區間：")
    for cond, range_info in boundaries.items():
        print(f"條件 [{cond}]:")
        print(f"- Cycle {range_info['start']} ~ {range_info['end']}")
        print(f"- 總數: {range_info['total_cycles']}")