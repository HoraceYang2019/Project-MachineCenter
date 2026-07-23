import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import initialize_config

# ====================== 主要功能：計算剩餘壽命 ======================
def calculate_remaining_life(csv_path, max_wear=0.3, pred_column='Final_Pred_c+2'):
    """
    讀取預測CSV，計算到達 max_wear 時的預估 Cycle 與剩餘壽命
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到預測檔案：{csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"已載入 {len(df)} 筆資料")
    
    # 清理資料：只保留有預測值的 row
    df_pred = df.dropna(subset=[pred_column]).copy()
    df_pred = df_pred.sort_values('Cycle')
    
    cycles = df_pred['Cycle'].values
    wear_pred = df_pred[pred_column].values
    
    # 找到第一個超過或接近 0.3mm 的點
    exceed_idx = np.where(wear_pred >= max_wear)[0]
    
    if len(exceed_idx) > 0:
        first_exceed_cycle = cycles[exceed_idx[0]]
        first_exceed_wear = wear_pred[exceed_idx[0]]
        print(f"\n預測刀具將在 Cycle {first_exceed_cycle} 達到 {first_exceed_wear:.4f} mm")
    else:
        # 若尚未達到，使用線性外插估計
        print(f"\n目前所有預測皆未達到 {max_wear}mm，使用線性外插估計...")
        # 使用最後兩點進行線性外插
        slope = (wear_pred[-1] - wear_pred[-2]) / (cycles[-1] - cycles[-2])
        if slope > 0:
            remaining_wear = max_wear - wear_pred[-1]
            extra_cycles = int(np.ceil(remaining_wear / slope))
            first_exceed_cycle = cycles[-1] + extra_cycles
            first_exceed_wear = max_wear
            print(f"線性外插估計：Cycle {first_exceed_cycle} 達到 {max_wear}mm")
        else:
            print("無法外插（磨耗趨勢異常）")
            return None
    
    # 計算剩餘壽命（相對於目前最後一個已知 cycle）
    current_max_cycle = int(df['Cycle'].max())
    remaining_cycles = first_exceed_cycle - current_max_cycle
    
    print(f"\n=== 剩餘壽命估計 ===")
    print(f"目前已加工 Cycle：{current_max_cycle}")
    print(f"預估壽命終止 Cycle：{first_exceed_cycle}")
    print(f"剩餘可加工 Cycle：{remaining_cycles} 個")
    print(f"剩餘壽命比例：{(remaining_cycles / first_exceed_cycle)*100:.1f}%")
    
    return {
        'current_cycle': current_max_cycle,
        'failure_cycle': int(first_exceed_cycle),
        'remaining_cycles': int(remaining_cycles),
        'max_wear': max_wear
    }


def plot_remaining_life(df, result_dict, output_path, max_wear=0.3):
    """繪製磨耗趨勢與壽命警示線"""
    plt.figure(figsize=(14, 8))
    
    plt.plot(df['Cycle'], df['Actual_Wear'], 's--', color='blue', linewidth=2, label='Actual Wear')
    plt.plot(df['Cycle'], df['Theoretical_Wear'], 'o-', color='green', linewidth=2, label='Theoretical')
    
    # 繪製各種預測 ['Final_Pred_i', 'Final_Pred_i+1', 'Final_Pred_i+2']
    # for col in ['Final_Pred_i', 'Final_Pred_i+1', 'Final_Pred_i+2']:
        # if col in df.columns:
    plt.plot(df['Cycle'], df['Final_Pred_c+2'], 'v--', color='red', linewidth=1.5, label='Model Prediction (C+2)')
    
    # 畫出 max_wear 失效線
    plt.axhline(y=max_wear, color='red', linestyle='--', linewidth=2, label=f'Max Allowable Wear ({max_wear}mm)')
    
    if result_dict:
        plt.axvline(x=result_dict['failure_cycle'], color='red', linestyle=':', 
                   label=f'Predicted End of Life (Cycle {result_dict["failure_cycle"]})')
    
    plt.xlabel('Machining Cycle')
    plt.ylabel('Tool Wear VB (mm)')
    plt.title('Tool Wear RUL Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(int(df['Cycle'].min()), result_dict["failure_cycle"] + 1, 1))
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    # print(f"圖表已儲存至：{output_path}")


# ====================== 主程式 ======================
def main():
    CONFIG = initialize_config('data_B')  # 可改成 paths_B 或 paths
    
    csv_path = os.path.join(CONFIG['paths']['result_dir'], 'fine_tune_prediction_NKUST-B.csv')
    # csv_path = os.path.join(CONFIG['result_dir'], 'fine_tune_prediction_QIT.csv')  # 若使用其他條件
    
    result = calculate_remaining_life(csv_path, max_wear=0.3)
    
    # 讀取原始資料畫圖
    df = pd.read_csv(csv_path)
    plot_path = os.path.join(CONFIG['paths']['visualization_dir'], 'RUL_estimation.png')
    plot_remaining_life(df, result, plot_path, max_wear=0.3)
    
    print("\n剩餘壽命分析完成！")


if __name__ == "__main__":
    main()