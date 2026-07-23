import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from glob import glob
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from config import initialize_config

start = time.time()

# GMM 偵測
def short_time_energy(data, window_size, step_size):
    """計算短時能量"""
    energies = []
    num_windows = (len(data) - window_size) // step_size + 1
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        energy = np.sum(np.square(data[start:end]))
        energies.append(energy)
    return np.array(energies).reshape(-1, 1)

def apply_gmm(energies, n_components=2, tol=1e-5):
    """套用 GMM 分類"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    energies_scaled = scaler.fit_transform(energies)
    gmm = GaussianMixture(
        n_components=n_components, 
        covariance_type='full',
        tol=tol, 
        random_state=0
    )
    gmm.fit(energies_scaled)
    return gmm.predict(energies_scaled)

def get_keep_indices(data, window_size=500, step_size=500):
    """回傳應保留的高能量區段索引"""
    energy_list = short_time_energy(data, window_size, step_size)
    labels = apply_gmm(energy_list)
    
    # 判斷哪一類是低能量
    low_energy_label = 1 if np.mean(energy_list[labels==1]) < np.mean(energy_list[labels==0]) else 0
    
    keep_indices = np.ones(len(data), dtype=bool)
    num_windows = (len(data) - window_size) // step_size + 1
    
    for i in range(num_windows):
        if labels[i] == low_energy_label:   # 低能量 → 移除
            start = i * step_size
            end = min(start + window_size, len(data))
            keep_indices[start:end] = False
            
    return keep_indices


def detect_segments_gmm(time_s, bending_data, torque_data, 
                       window_size=500, step_size=500, min_duration_sec=0.5):
    """
    使用 Bending 訊號做 GMM 能量判斷，保留對應的 Torque 區段
    """
    # 1. 對 Bending 計算能量並做 GMM
    energy_list = short_time_energy(bending_data, window_size, step_size)
    labels = apply_gmm(energy_list)
    
    # 判斷哪一群是低能量（非加工）
    low_energy_label = 1 if np.mean(energy_list[labels==1]) < np.mean(energy_list[labels==0]) else 0
    
    # 2. 建立保留遮罩（基於 Bending 的判斷）
    keep_indices = np.ones(len(bending_data), dtype=bool)
    num_windows = (len(bending_data) - window_size) // step_size + 1
    
    for i in range(num_windows):
        if labels[i] == low_energy_label:        # 低能量 → 移除
            start = i * step_size
            end = min(start + window_size, len(bending_data))
            keep_indices[start:end] = False
    
    # 3. 提取連續加工區段
    segments = []
    in_segment = False
    s_idx = 0
    
    for i in range(len(keep_indices)):
        if keep_indices[i] and not in_segment:
            s_idx = i
            in_segment = True
        elif not keep_indices[i] and in_segment:
            e_idx = i - 1
            duration = time_s[e_idx] - time_s[s_idx]
            if duration >= min_duration_sec:
                segments.append({
                    'start_idx': s_idx,
                    'end_idx': e_idx,
                    'duration': float(duration),
                    'start_time': float(time_s[s_idx]),
                    'end_time': float(time_s[e_idx])
                })
            in_segment = False

    if in_segment:
        e_idx = len(keep_indices) - 1
        duration = time_s[e_idx] - time_s[s_idx]
        if duration >= min_duration_sec:
            segments.append({
                'start_idx': s_idx,
                'end_idx': e_idx,
                'duration': float(duration),
                'start_time': float(time_s[s_idx]),
                'end_time': float(time_s[e_idx])
            })

    # 4. 用於畫圖的 Torque 訊號（非加工區設為 NaN）
    viz_torque = np.where(keep_indices, torque_data, np.nan)
    
    return segments, viz_torque, keep_indices


def detect_machining_segments(time_s, torque, rolling_window, min_duration_sec):
    """偵測多個獨立的加工區段（穩定版）"""

    # 計算平滑序列 sm(t) = median(Tq(t-25), ..., Tq(t+25))
    rolling_median = pd.Series(torque).rolling(
        window=rolling_window, center=True, min_periods=1
    ).median().values

    # 取平滑序列的最大值與最小值的平均作為門檻
    sm_max = np.max(rolling_median)
    sm_min = np.min(rolling_median)
    thresh = (sm_max + sm_min) / 2

    # 使用這個門檻產生遮罩
    mask = rolling_median > thresh

    # 找出連續區段
    segments = []
    in_segment = False
    start = 0

    for i in range(len(mask)):
        if mask[i] and not in_segment:
            start = i
            in_segment = True
        elif not mask[i] and in_segment:
            end = i - 1
            duration = time_s[end] - time_s[start]
            if duration >= min_duration_sec:
                segments.append({
                    'start_idx': start,
                    'end_idx': end,
                    'duration': float(duration),
                    'start_time': float(time_s[start]),
                    'end_time': float(time_s[end])
                })
            in_segment = False

    if in_segment:
        end = len(mask) - 1
        duration = time_s[end] - time_s[start]
        if duration >= min_duration_sec:
            segments.append({
                'start_idx': start,
                'end_idx': end,
                'duration': float(duration),
                'start_time': float(time_s[start]),
                'end_time': float(time_s[end])
            })

    return segments, rolling_median, thresh


def main(path_key):
    
    CONFIG = initialize_config(path_key)

    data_root = CONFIG[path_key]['data_root']
    output_dir = CONFIG['paths']['pkl_dir']
    viz_dir = CONFIG['paths']['visualization_dir']
    # suffix = CONFIG[path_key]['data_root'].split('-')[-1]
    affix = CONFIG[path_key]['data_root'].lstrip('./\\')
    
    print(f"Data processing for {affix}\n")
    
    csv_paths = sorted(
        glob(os.path.join(data_root, '*', 'fastdtw.csv')),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).replace('cycle', ''))
    )

    processed = {}

    for idx, csv_path in enumerate(csv_paths, 1):
        folder_name = os.path.basename(os.path.dirname(csv_path))
        cycle_num = int(folder_name.replace('cycle', ''))
        cycle_key = f"cycle_{cycle_num}"

        df = pd.read_csv(csv_path)
        bending = df['BendingX'].values
        torque = df['Torque'].values
        time_s = df['Time_s'].values

        segments, viz_signal, keep_mask = detect_segments_gmm(
            time_s, bending, torque,
            window_size=CONFIG['data_processing']['window_size'],
            step_size=CONFIG['data_processing']['step_size'],
            min_duration_sec=CONFIG['data_processing']['min_duration_sec']
        )
        thresh = None
        
        if not segments:
            print(f"Cycle {cycle_num} has no valid machining segments")
            continue

        cycle_segments = []
        cycle_total_duration = 0.0

        for seg_idx, seg in enumerate(segments):
            seg_df = df.iloc[seg['start_idx']:seg['end_idx']+1].reset_index(drop=True)
            
            segment_data = {
                'segment_id': seg_idx,
                'start_idx': seg['start_idx'],
                'end_idx': seg['end_idx'],
                'duration': seg['duration'],
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'segment_df': seg_df.to_dict('list'),   # 重要：保留給 theoretical_estimation
                'num_windows': 1                         # 不產生 sliding window
            }
            
            cycle_segments.append(segment_data)
            cycle_total_duration += seg['duration']

        processed[cycle_key] = {
            'segments': cycle_segments,
            'total_machining_duration': float(cycle_total_duration),
            'total_record_duration': float(time_s[-1] - time_s[0]),
            'num_segments': len(cycle_segments)
        }

        print(f"Cycle {cycle_num:2d} → {len(cycle_segments)} 個加工區段 | 時間 {cycle_total_duration:.2f} s")

        # Cycle 1 視覺化
        # if idx in [1]:
        plt.figure(figsize=(16, 9))
        plt.plot(time_s, torque, label='Raw Torque', color='blue', alpha=0.7)
        plt.plot(time_s, viz_signal, label='GMM Energy Mask' , color='red', linewidth=2)

        for i, seg in enumerate(segments):
            color = plt.cm.tab10(i % 10)
            plt.axvspan(seg['start_time'], seg['end_time'], alpha=0.25, color=color)
            plt.text((seg['start_time'] + seg['end_time'])/2, np.max(torque)*1.0,
                        f'Seg {i+1}\n{seg["duration"]:.1f}s', ha='center', color='red', fontsize=9)
        if thresh is not None:
            plt.axhline(y=thresh, color='green', linestyle='--', label=f'Threshold')
        plt.title(f'{affix}_Cycle {cycle_num}_Detected Machining Segments (GMM Energy)')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque')
        plt.legend()
        plt.grid(True, alpha=0.3)

        os.makedirs(os.path.join(viz_dir, affix), exist_ok=True)
        plt.savefig(os.path.join(viz_dir, affix, f'{affix}_cycle_{cycle_num:02d}_processed.png'), dpi=300, bbox_inches='tight')
        plt.close()

    with open(os.path.join(output_dir, f"{CONFIG['paths']['processed_data']}_{affix}.pkl"), 'wb') as f:
        torch.save(processed, f)
    
    total_cycles = len(processed)
    print("\n資料處理完成！")
    print(f"處理 Cycle 總數   : {total_cycles} 個")
    print("\nData Processing completed!")
    print(f"執行時間: {time.time() - start:.2f} 秒")

if __name__ == "__main__":
    main('data_AB')