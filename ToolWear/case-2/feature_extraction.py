import os
import torch
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
from scipy.fft import rfft
from config import initialize_config

start = time.time()

def main(path_key):
    CONFIG = initialize_config(path_key)
    # suffix = CONFIG[path_key]['data_root'].split('-')[-1]
    affix = CONFIG[path_key]['data_root'].lstrip('./\\')

    print(f"Feature extraction for {affix}\n")

    processed_path = os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['processed_data']}_{affix}.pkl")
    # 直接將 pkl_dir、檔名、字尾與副檔名串接
    with open(processed_path, 'rb') as f:
        processed = torch.load(f)
    print(f"資料載入完成，耗時 {(time.time() - start):.2f} 秒")

    # ==================== 加工參數取得（支援有/無 conditions）====================
    conditions = CONFIG['wear_data'].get('conditions', {})
    indexed_cycles = CONFIG['wear_data'].get('indexed_cycles', {})
    
    # 單一條件 fallback（config 最上層 machining_parameter）
    default_mach_param = CONFIG.get('machining_parameter', {}) or \
                        CONFIG.get('wear_data', {}).get('machining_parameter', {})

    condition_params = {}
    if conditions:
        condition_params = {cond: data.get('machining_parameter', {}) 
                           for cond, data in conditions.items()}
    else:
        # 沒有 conditions 時全部使用 default
        condition_params = {"1": default_mach_param}   # 預設 condition = "1"

    print(f"使用 {'多條件' if conditions else '單一條件（default）'} 模式")


    # ==================== 1. 加工參數正規化（Min-Max）====================
    param_ranges = {
        'feed_rate': (50, 1000),
        'cutting_speed': (10, 500),
        'ap': (0.1, 10),
        'ae': (0.1, 10)
    }

    print("Starting cycle-level feature extraction...")

    all_base_features = []
    all_mach_params = [] # 儲存每個 cycle 的 mach_params
    cycle_keys = []
    cycle_conditions = [] # 記錄每個 cycle 的 condition
    
    for cycle_key, data in processed.items():
        cycle_num = int(cycle_key.split('_')[1])

        # 取得 condition（有 indexed_cycles 就用，否則預設 "1"）
        cond = indexed_cycles.get(str(cycle_num), {}).get("condition", "1")
        cycle_conditions.append(cond)
        mach_param = condition_params.get(cond, default_mach_param)
        
        # 每個 cycle 獨立正規化加工參數
        mach_params = np.array([
            (mach_param['feed_rate'] - param_ranges['feed_rate'][0]) /
            (param_ranges['feed_rate'][1] - param_ranges['feed_rate'][0]),
            (mach_param['cutting_speed'] - param_ranges['cutting_speed'][0]) /
            (param_ranges['cutting_speed'][1] - param_ranges['cutting_speed'][0]),
            (mach_param['ap'] - param_ranges['ap'][0]) /
            (param_ranges['ap'][1] - param_ranges['ap'][0]),
            (mach_param['ae'] - param_ranges['ae'][0]) /
            (param_ranges['ae'][1] - param_ranges['ae'][0])
        ], dtype=np.float32)

        all_mach_params.append(mach_params)

        # ==================== Torque 特徵 ====================
        all_torque = []
        for seg in data.get('segments', []):
            seg_df = pd.DataFrame(seg['segment_df'])
            all_torque.extend(seg_df['Torque'].values)

        torque = np.array(all_torque, dtype=np.float32)

        # 頻域特徵
        xf = rfft(torque)
        pk = (np.abs(xf)**2) / len(torque) # 功率譜
        sum_pk = np.sum(pk)
        e_k = np.log(1 + sum_pk) # 頻譜能量
        p_k_dist = pk / (sum_pk) # 正規化能量分布
        h_k = -np.sum(p_k_dist * np.log(p_k_dist)) # 頻譜熵

        base_feat = np.array([
            np.max(torque),
            np.sqrt(np.mean(torque**2)), # RMS
            # np.mean(torque),
            # np.std(torque),
            # np.min(torque),
            # np.median(torque), # 中位數 (Median)
            # skew(torque), # 偏態 (Skewness)
            # kurtosis(torque), # 峰度 (Kurtosis)
            # e_k,
            # h_k,
            # np.sum(np.abs(rfft(torque))**2) # FFT Energy
        ], dtype=np.float32)

        all_base_features.append(base_feat)
        cycle_keys.append(cycle_key)

    all_base_features = np.array(all_base_features)

    # ==================== 3. 對 Torque 特徵做 MinMaxScaler ====================
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_base = scaler.fit_transform(all_base_features)   # shape: (n_cycles, 6)

    print(f"MinMaxScaler 完成，所有 Torque 特徵已縮放到 [0, 1] 範圍")

    # ==================== 4. 合併特徵並儲存 ====================
    cycle_features = {}
    for i, cycle_key in enumerate(cycle_keys):
        combined = np.concatenate([normalized_base[i], all_mach_params[i]])
        cycle_features[cycle_key] = [combined]

    # 儲存特徵
    features_path = os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['extracted_features']}_{affix}.pkl")
    with open(features_path, 'wb') as f:
        torch.save(cycle_features, f)

    print(f"Feature Extraction 完成")
    print(f"總 Cycle 數: {len(cycle_features)}")
    print(f"每個 cycle 特徵維度: {len(combined)}")
    print(f"執行時間: {time.time() - start:.2f} 秒")

    # 儲存 CSV
    csv_data = []
    feature_columns = ['max', 'rms', 'feed_rate','cutting_speed', 'ap', 'ae']

    for i, cycle_key in enumerate(cycle_keys):
        cond = cycle_conditions[i]
        combined = np.concatenate([normalized_base[i], all_mach_params[i]])
        csv_row = [cycle_key, cond] + combined.tolist()
        csv_data.append(csv_row)

    df_features = pd.DataFrame(csv_data, columns=['cycle_id', 'condition'] + feature_columns)
    csv_output_path = os.path.join(CONFIG['paths']['result_dir'], f"{CONFIG['paths']['extracted_features']}_{affix}.csv")
    df_features.to_csv(csv_output_path, index=False)

if __name__ == "__main__":
    main('data_AB')