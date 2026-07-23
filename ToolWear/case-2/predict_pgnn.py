import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from train_pgnn import MLP
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from config import initialize_config, add_condition_background

def main(path_key):
    print("=== Prediction Phase: MLP Residual Prediction ===")

    CONFIG = initialize_config(path_key)
    # suffix = CONFIG[path_key]['data_root'].split('-')[-1]
    affix = CONFIG[path_key]['data_root'].lstrip('./\\')

    # 載入模型
    model_path = os.path.join(CONFIG['paths']['model_dir'], 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型，請先執行 train_mlp.py")

    # 載入資料
    features_path = os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['extracted_features']}_{affix}.pkl")
    cycle_features = torch.load(features_path, weights_only=False)

    theo_path = os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['theoretical_estimation']}_{affix}.pkl")
    theo_data = torch.load(theo_path, weights_only=False)

    cycle_list = sorted(cycle_features.keys(), key=lambda k: int(k.split('_')[1]))
    
    print(f"資料集類型: 混合條件資料")
    print(f"總 Cycle 數: {len(cycle_list)}")

    # ==================== 完整 Actual & Theoretical ====================
    all_cycles = []
    all_actual = []
    all_theo = []

    for ck in cycle_list:
        cnum = int(ck.split('_')[1])
        all_cycles.append(cnum)
        all_actual.append(CONFIG['actual_wear'].get(cnum, 0.0))
        all_theo.append(theo_data['cumulative_vb'].get(ck, 0.0))

    # ==================== 建立預測輸入 ====================
    X_list = []
    pred_base_cycles = []        # 當前 cycle (i)

    for i in range(1, len(cycle_list) - 2):
        prev_feat = cycle_features[cycle_list[i-1]][0]
        curr_feat = cycle_features[cycle_list[i]][0]
        input_feat = np.concatenate([prev_feat, curr_feat])
        
        X_list.append(input_feat)
        pred_base_cycles.append(cycle_list[i])

    X = torch.tensor(np.array(X_list, dtype=np.float32))

    # ==================== 模型預測 ====================
    input_dim = X.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=CONFIG['model']['hidden_dim'])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 建立 theo_matrix
    theo_matrix = []
    theo_i0 = []
    theo_i1 = []
    theo_i2 = []

    for ck in pred_base_cycles:
        cnum = int(ck.split('_')[1])
        t0 = theo_data['cumulative_vb'].get(f"cycle_{cnum}", 0.0)
        t1 = theo_data['cumulative_vb'].get(f"cycle_{cnum + 1}", 0.0)
        t2 = theo_data['cumulative_vb'].get(f"cycle_{cnum + 2}", 0.0)
        
        theo_matrix.append([t0, t1, t2])
        theo_i0.append(t0)
        theo_i1.append(t1)
        theo_i2.append(t2)
    
    X_theo_tensor = torch.tensor(theo_matrix, dtype=torch.float32)

    with torch.no_grad():
        pred_residuals = model(X, X_theo_tensor).numpy()

    pred_i0 = pred_residuals[:, 0]
    pred_i1 = pred_residuals[:, 1]
    pred_i2 = pred_residuals[:, 2]

    final_pred_i0 = np.array(theo_i0) + pred_i0
    final_pred_i1 = np.array(theo_i1) + pred_i1
    final_pred_i2 = np.array(theo_i2) + pred_i2

    pred_cycle_i0 = [int(ck.split('_')[1]) for ck in pred_base_cycles]
    pred_cycle_i1 = [int(ck.split('_')[1]) + 1 for ck in pred_base_cycles]
    pred_cycle_i2 = [int(ck.split('_')[1]) + 2 for ck in pred_base_cycles]

    # ==================== 評估 ====================
    def evaluate(pred_cycle, final_pred):
        y_act = []
        y_pred = []
        for c, p in zip(pred_cycle, final_pred):
            if c in all_cycles:
                idx = all_cycles.index(c)
                y_act.append(all_actual[idx])
                y_pred.append(p)
        return np.array(y_act), np.array(y_pred)

    y_act_i0, y_pred_i0 = evaluate(pred_cycle_i0, final_pred_i0)
    y_act_i1, y_pred_i1 = evaluate(pred_cycle_i1, final_pred_i1)
    y_act_i2, y_pred_i2 = evaluate(pred_cycle_i2, final_pred_i2)

    print(f"\n=== c 步預測 ===")
    print(f"MAE  = {mean_absolute_error(y_act_i0, y_pred_i0):.4f} mm")
    print(f"MAPE = {mean_absolute_percentage_error(y_act_i0, y_pred_i0) * 100:.2f} %")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_act_i0, y_pred_i0)):.4f} mm")
    print(f"R²   = {r2_score(y_act_i0, y_pred_i0):.4f}")

    print(f"\n=== c+1 步預測 ===")
    print(f"MAE  = {mean_absolute_error(y_act_i1, y_pred_i1):.4f} mm")
    print(f"MAPE = {mean_absolute_percentage_error(y_act_i1, y_pred_i1) * 100:.2f} %")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_act_i1, y_pred_i1)):.4f} mm")
    print(f"R²   = {r2_score(y_act_i1, y_pred_i1):.4f}")

    print(f"\n=== c+2 步預測 ===")
    print(f"MAE  = {mean_absolute_error(y_act_i2, y_pred_i2):.4f} mm")
    print(f"MAPE = {mean_absolute_percentage_error(y_act_i2, y_pred_i2) * 100:.2f} %")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_act_i2, y_pred_i2)):.4f} mm")
    print(f"R²   = {r2_score(y_act_i2, y_pred_i2):.4f}")

    # 繪製比較圖
    plt.figure(figsize=(18, 8))
    
    # === 加入條件背景色塊 ===
    ax = plt.gca()

    plt.plot(all_cycles, all_actual, 's--', color='blue', linewidth=2, label='Actual')
    plt.plot(all_cycles, all_theo, 'o-', color='green', linewidth=2, label='Theoretical')
    
    plt.plot(pred_cycle_i0, final_pred_i0, 'x--', color='purple', linewidth=1.5, label='Predicted (c)')
    plt.plot(pred_cycle_i1, final_pred_i1, '^--', color='orange', linewidth=1.5, label='Predicted (c+1)')
    plt.plot(pred_cycle_i2, final_pred_i2, 'v--', color='red', linewidth=1.5, label='Predicted (c+2)')

    plt.xlabel('Machining Cycle', fontsize=20)
    plt.ylabel('Tool Wear VB (mm)', fontsize=20)
    plt.title(f'{affix} PGNN Predicted Tool Wear', fontsize=24, fontweight='bold')

    # 設定好 Y 軸極限後再畫背景
    ax.set_ylim(bottom=0)
    # === 加入條件背景色塊 ===
    bg_patches = add_condition_background(ax)

    # === 圖例處理 ===
    legend_handles = [
        Line2D([0], [0], color='blue', marker='s', linestyle='--', linewidth=2, label='Actual Wear'),
        Line2D([0], [0], color='green', marker='o', linestyle='-', linewidth=2, label='Theoretical Wear'),
        Line2D([0], [0], color='purple', marker='x', linestyle='--', linewidth=1.5, label='Predicted (C)'),
        Line2D([0], [0], color='orange', marker='^', linestyle='--', linewidth=1.5, label='Predicted (C+1)'),
        Line2D([0], [0], color='red', marker='v', linestyle='--', linewidth=1.5, label='Predicted (C+2)')
    ]
    plt.legend(handles=legend_handles, fontsize=14, loc='upper left')

    plt.grid(True, which='major', linestyle='-', alpha=0.7)    
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5)) # 平均切成 5 等份
    plt.grid(True, which='minor', linestyle=':', alpha=0.7) # 次要格線
    plt.xticks(range(min(all_cycles), max(all_cycles)+1, 2))
    plt.tick_params(axis='both', which='major', labelsize=16)

    ax.set_xlim(left=min(all_cycles) - 1, right=max(all_cycles) + 1)
    ax.set_ylim(bottom=0)

    plot_path = os.path.join(CONFIG['paths']['visualization_dir'], f'{affix}_pgnn_prediction.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    # ==================== 儲存 CSV ====================
    results = []
    for c in sorted(set(all_cycles + pred_cycle_i1 + pred_cycle_i2)):
        row = {'Cycle': c}
        
        if c in all_cycles:
            idx = all_cycles.index(c)
            row['Actual_Wear'] = all_actual[idx]
            row['Theoretical_Wear'] = all_theo[idx]
        else:
            row['Actual_Wear'] = np.nan
            row['Theoretical_Wear'] = np.nan

        if c in pred_cycle_i0:
            idx = pred_cycle_i0.index(c)
            row['Pred_Residual_c'] = pred_i0[idx]
            row['Final_Pred_c'] = final_pred_i0[idx]
        else:
            row['Pred_Residual_c'] = np.nan
            row['Final_Pred_c'] = np.nan

        if c in pred_cycle_i1:
            idx = pred_cycle_i1.index(c)
            row['Pred_Residual_c+1'] = pred_i1[idx]
            row['Final_Pred_c+1'] = final_pred_i1[idx]
        else:
            row['Pred_Residual_c+1'] = np.nan
            row['Final_Pred_c+1'] = np.nan
        
        if c in pred_cycle_i2:
            idx = pred_cycle_i2.index(c)
            row['Pred_Residual_c+2'] = pred_i2[idx]
            row['Final_Pred_c+2'] = final_pred_i2[idx]
        else:
            row['Pred_Residual_c+2'] = np.nan
            row['Final_Pred_c+2'] = np.nan
        
        results.append(row)

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(CONFIG['paths']['result_dir'], f'prediction_{affix}.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8', float_format='%.6f')

    print(f"預測結果已儲存至: {csv_path}")

    # ==================== 產生漂亮的 Predicted VS Actual Wear Summary CSV ====================
    summary_data = {
        'Metric': ['MAE', 'RMSE', 'R²', 'MAPE'],
        'C-Cycle': [
            mean_absolute_error(y_act_i0, y_pred_i0),
            np.sqrt(mean_squared_error(y_act_i0, y_pred_i0)),
            r2_score(y_act_i0, y_pred_i0),
            mean_absolute_percentage_error(y_act_i0, y_pred_i0) * 100
        ],
        'C+1 Cycle': [
            mean_absolute_error(y_act_i1, y_pred_i1),
            np.sqrt(mean_squared_error(y_act_i1, y_pred_i1)),
            r2_score(y_act_i1, y_pred_i1),
            mean_absolute_percentage_error(y_act_i1, y_pred_i1) * 100
        ],
        'C+2 Cycle': [
            mean_absolute_error(y_act_i2, y_pred_i2),
            np.sqrt(mean_squared_error(y_act_i2, y_pred_i2)),
            r2_score(y_act_i2, y_pred_i2),
            mean_absolute_percentage_error(y_act_i2, y_pred_i2) * 100
        ],
        'Unit': ['mm', 'mm', '', '%']
    }

    summary_df = pd.DataFrame(summary_data)
    
    # 格式化顯示（可選）
    summary_df['C-Cycle'] = summary_df['C-Cycle'].map('{:.4f}'.format)
    summary_df['C+1 Cycle'] = summary_df['C+1 Cycle'].map('{:.4f}'.format)
    summary_df['C+2 Cycle'] = summary_df['C+2 Cycle'].map('{:.4f}'.format)
    summary_df.loc[summary_df['Metric'] == 'MAPE', ['C-Cycle', 'C+1 Cycle', 'C+2 Cycle']] = \
        summary_df.loc[summary_df['Metric'] == 'MAPE', ['C-Cycle', 'C+1 Cycle', 'C+2 Cycle']].map('{}'.format)

    summary_path = os.path.join(CONFIG['paths']['result_dir'], f'{affix}_prediction_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Prediction Summary 已儲存至: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main('data_AB')