import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from config import initialize_config, add_condition_background
from train_pgnn import MLP

def main(path_key='data_AB', fine_tune_samples=3, fine_tune_epochs=800, fine_tune_lr=0.0005):
    print("=== Fine-Tune + Prediction Phase: PGNN MLP Residual ===\n")
    print(f"使用前 {fine_tune_samples} 筆 cycle 進行 Fine-tuning")

    CONFIG = initialize_config(path_key)
    affix = CONFIG[path_key]['data_root'].lstrip('./\\')

    # ==================== 載入模型與資料 ====================
    model_path = os.path.join(CONFIG['paths']['model_dir'], 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到預訓練模型 {model_path}，請先執行 train_pgnn.py")

    features_path = os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['extracted_features']}_{affix}.pkl")
    cycle_features = torch.load(features_path, weights_only=False)

    theo_path = os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['theoretical_estimation']}_{affix}.pkl")
    theo_data = torch.load(theo_path, weights_only=False)

    cycle_list = sorted(cycle_features.keys(), key=lambda k: int(k.split('_')[1]))

    # 全資料 Actual & Theoretical
    all_cycles = []
    all_actual = []
    all_theo = []
    for ck in cycle_list:
        cnum = int(ck.split('_')[1])
        all_cycles.append(cnum)
        all_actual.append(CONFIG['actual_wear'].get(cnum, 0.0))
        all_theo.append(theo_data['cumulative_vb'].get(ck, 0.0))

    print(f"總 Cycle 數: {len(cycle_list)} | Fine-tune 使用前 {fine_tune_samples} 筆")

    if len(cycle_list) < fine_tune_samples + 3:
        raise ValueError(f"資料不足，至少需要 {fine_tune_samples + 3} 筆 cycle")

    # ==================== Fine-Tuning 資料準備 ====================
    X_ft, y_ft, theo_ft = [], [], []

    for i in range(1, fine_tune_samples + 1):
        if i + 2 >= len(cycle_list):
            break
        prev_feat = cycle_features[cycle_list[i-1]][0]
        curr_feat = cycle_features[cycle_list[i]][0]
        input_feat = np.concatenate([prev_feat, curr_feat])
        X_ft.append(input_feat)

        c_i = int(cycle_list[i].split('_')[1])
        res_i   = CONFIG['actual_wear'].get(c_i, 0.0) - theo_data['cumulative_vb'].get(f"cycle_{c_i}", 0.0)
        res_ip1 = CONFIG['actual_wear'].get(c_i+1, 0.0) - theo_data['cumulative_vb'].get(f"cycle_{c_i+1}", 0.0)
        res_ip2 = CONFIG['actual_wear'].get(c_i+2, 0.0) - theo_data['cumulative_vb'].get(f"cycle_{c_i+2}", 0.0)

        y_ft.append([res_i, res_ip1, res_ip2])
        theo_ft.append([
            theo_data['cumulative_vb'].get(f"cycle_{c_i}", 0.0),
            theo_data['cumulative_vb'].get(f"cycle_{c_i+1}", 0.0),
            theo_data['cumulative_vb'].get(f"cycle_{c_i+2}", 0.0)
        ])

    X_ft = torch.tensor(np.array(X_ft, dtype=np.float32))
    y_ft = torch.tensor(np.array(y_ft, dtype=np.float32))
    theo_ft = torch.tensor(np.array(theo_ft, dtype=np.float32))

    # ==================== Fine-Tuning ====================
    model = MLP(input_dim=X_ft.shape[1], hidden_dim=CONFIG['model']['hidden_dim'])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
    lambda_phys = CONFIG['model']['lambda_phys']

    print(f"開始 Fine-tune {fine_tune_epochs} epochs (lr={fine_tune_lr})...")

    model.train()
    best_loss = float('inf')
    for epoch in range(fine_tune_epochs):
        optimizer.zero_grad()
        
        pred_residual = model(X_ft, theo_ft)
        
        loss_data = nn.MSELoss()(pred_residual, y_ft)
        loss_phys = nn.MSELoss()(pred_residual, torch.zeros_like(pred_residual))

        final_pred = theo_ft + pred_residual
        loss_mono = torch.mean(torch.relu(final_pred[:, 0] - final_pred[:, 1])) + \
                    torch.mean(torch.relu(final_pred[:, 1] - final_pred[:, 2]))

        loss = (1 - lambda_phys) * loss_data + lambda_phys * loss_phys + 1 * loss_mono

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == fine_tune_epochs - 1:
            print(f"Epoch {epoch+1:4d}/{fine_tune_epochs} | Loss: {loss.item():.2e} | "
                  f"Data: {loss_data.item():.2e} | Phys: {loss_phys.item():.2e}")

    print(f"Fine-tune 完成！Best Loss: {best_loss:.2e}")

    # ==================== 後續預測 ====================
    model.eval()
    X_list = []
    pred_base_cycles = []

    start_idx = fine_tune_samples
    for i in range(start_idx, len(cycle_list) - 2):
        prev_feat = cycle_features[cycle_list[i-1]][0]
        curr_feat = cycle_features[cycle_list[i]][0]
        X_list.append(np.concatenate([prev_feat, curr_feat]))
        pred_base_cycles.append(cycle_list[i])

    X = torch.tensor(np.array(X_list, dtype=np.float32))

    # theo matrix
    theo_matrix = []
    theo_i0 = []
    theo_i1 = []
    theo_i2 = []

    for ck in pred_base_cycles:
        cnum = int(ck.split('_')[1])
        t0 = theo_data['cumulative_vb'].get(f"cycle_{cnum}", 0.0)
        t1 = theo_data['cumulative_vb'].get(f"cycle_{cnum+1}", 0.0)
        t2 = theo_data['cumulative_vb'].get(f"cycle_{cnum+2}", 0.0)
        theo_matrix.append([t0, t1, t2])
        theo_i0.append(t0)
        theo_i1.append(t1)
        theo_i2.append(t2)

    X_theo = torch.tensor(theo_matrix, dtype=torch.float32)

    with torch.no_grad():
        pred_residuals = model(X, X_theo).numpy()

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
        y_act = [all_actual[all_cycles.index(c)] for c in pred_cycle if c in all_cycles]
        y_pred = [p for c, p in zip(pred_cycle, final_pred) if c in all_cycles]
        return np.array(y_act), np.array(y_pred)

    y_act_i0, y_pred_i0 = evaluate(pred_cycle_i0, final_pred_i0)
    y_act_i1, y_pred_i1 = evaluate(pred_cycle_i1, final_pred_i1)
    y_act_i2, y_pred_i2 = evaluate(pred_cycle_i2, final_pred_i2)

    print(f"\n=== Fine-tune 後 c 步預測 ===")
    print(f"MAE  = {mean_absolute_error(y_act_i0, y_pred_i0):.4f} mm")
    print(f"MAPE = {mean_absolute_percentage_error(y_act_i0, y_pred_i0)*100:.2f} %")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_act_i0, y_pred_i0)):.4f} mm")
    print(f"R²   = {r2_score(y_act_i0, y_pred_i0):.4f}")

    print(f"\n=== c+1 步預測 ===")
    print(f"MAE  = {mean_absolute_error(y_act_i1, y_pred_i1):.4f} mm")
    print(f"MAPE = {mean_absolute_percentage_error(y_act_i1, y_pred_i1)*100:.2f} %")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_act_i1, y_pred_i1)):.4f} mm")
    print(f"R²   = {r2_score(y_act_i1, y_pred_i1):.4f}")

    print(f"\n=== c+2 步預測 ===")
    print(f"MAE  = {mean_absolute_error(y_act_i2, y_pred_i2):.4f} mm")
    print(f"MAPE = {mean_absolute_percentage_error(y_act_i2, y_pred_i2)*100:.2f} %")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_act_i2, y_pred_i2)):.4f} mm")
    print(f"R²   = {r2_score(y_act_i2, y_pred_i2):.4f}")

    # ==================== 繪圖 ====================
    plt.figure(figsize=(18, 8))
    ax = plt.gca()

    plt.plot(all_cycles, all_actual, 's--', color='blue', linewidth=2, label='Actual')
    plt.plot(all_cycles, all_theo, 'o-', color='green', linewidth=2, label='Theoretical')
    
    plt.plot(pred_cycle_i0, final_pred_i0, 'x--', color='purple', linewidth=1.5, label='Predicted (C)')
    plt.plot(pred_cycle_i1, final_pred_i1, '^--', color='orange', linewidth=1.5, label='Predicted (C+1)')
    plt.plot(pred_cycle_i2, final_pred_i2, 'v--', color='red', linewidth=1.5, label='Predicted (C+2)')

    add_condition_background(ax)   # 使用 config.py 中的動態版本

    plt.xlabel('Machining Cycle', fontsize=18)
    plt.ylabel('Tool Wear VB (mm)', fontsize=18)
    plt.title(f'{affix} PGNN Fine-Tune Prediction', fontsize=22, fontweight='bold')

    legend_handles = [
        Line2D([0], [0], color='blue', marker='s', linestyle='--', linewidth=2, label='Actual Wear'),
        Line2D([0], [0], color='green', marker='o', linestyle='-', linewidth=2, label='Theoretical Wear'),
        Line2D([0], [0], color='purple', marker='x', linestyle='--', linewidth=1.5, label='Predicted (C)'),
        Line2D([0], [0], color='orange', marker='^', linestyle='--', linewidth=1.5, label='Predicted (C+1)'),
        Line2D([0], [0], color='red', marker='v', linestyle='--', linewidth=1.5, label='Predicted (C+2)')
    ]
    plt.legend(handles=legend_handles, fontsize=13, loc='upper left')

    plt.grid(True, which='major', linestyle='-', alpha=0.7)
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    plt.grid(True, which='minor', linestyle=':', alpha=0.7)
    plt.xticks(range(min(all_cycles), max(all_cycles)+1, 2))
    plt.tick_params(axis='both', labelsize=14)

    plot_path = os.path.join(CONFIG['paths']['visualization_dir'], f'{affix}_pgnn_fine_tune_prediction.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"預測圖已儲存：{plot_path}")

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
    csv_path = os.path.join(CONFIG['paths']['result_dir'], f'fine_tune_prediction_{affix}.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8', float_format='%.6f')
    
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

    summary_path = os.path.join(CONFIG['paths']['result_dir'], f'{affix}_fine_tune_prediction_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Prediction Summary 已儲存至: {summary_path}")
    print(summary_df.to_string(index=False))

    print("\n=== Fine-Tune + Prediction 完成 ===")


if __name__ == "__main__":
    main(path_key='data_AB', 
         fine_tune_samples=3,      # 可調整
         fine_tune_epochs=800, 
         fine_tune_lr=0.0005)