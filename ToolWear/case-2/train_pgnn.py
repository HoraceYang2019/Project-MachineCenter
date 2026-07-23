import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import initialize_config

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 3)
        )
    
    def forward(self, x, theo_val):
        # 1. 原始殘差預測
        raw_residual = self.net(x)
        
        # 2. 計算最終預測值
        raw_final_pred = theo_val + raw_residual
        
        # 3. 使用 Softplus 確保最終磨耗值 >= 0
        final_pred = torch.nn.functional.softplus(raw_final_pred)
        
        # 4. 反推安全的殘差
        safe_residual = final_pred - theo_val
        return safe_residual


def main(path_key):
    print("=== Modeling Phase: Physics-Informed MLP ===")

    global CONFIG
    CONFIG = initialize_config(path_key)
    # suffix = CONFIG[path_key]['data_root'].split('-')[-1]
    affix = CONFIG[path_key]['data_root'].lstrip('./\\')

    # 載入資料
    features_path = os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['extracted_features']}_{affix}.pkl")
    cycle_features = torch.load(features_path, weights_only=False)

    theo_path = os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['theoretical_estimation']}_{affix}.pkl")
    theo_data = torch.load(theo_path, weights_only=False)

    # ==================== 直接使用原始資料建立訓練樣本 ====================
    cycle_list = sorted(cycle_features.keys(), key=lambda k: int(k.split('_')[1]))
    
    print(f"原始資料筆數: {len(cycle_list)} 筆")
    print(f"資料集類型: 混合條件資料")

    x_list = []
    y_list = []
    theo_steps_list = []

    for i in range(1, len(cycle_list) - 2):
        # 輸入特徵 = [前一筆, 當前筆]
        prev_feat = cycle_features[cycle_list[i-1]][0]
        curr_feat = cycle_features[cycle_list[i]][0]
        input_feat = np.concatenate([prev_feat, curr_feat])
        x_list.append(input_feat)

        # 真實殘差 [i, i+1, i+2]
        c_i = int(cycle_list[i].split('_')[1])
        res_i   = CONFIG['actual_wear'].get(c_i, 0.0)     - theo_data['cumulative_vb'].get(f"cycle_{c_i}", 0.0)
        res_ip1 = CONFIG['actual_wear'].get(c_i+1, 0.0)   - theo_data['cumulative_vb'].get(f"cycle_{c_i+1}", 0.0)
        res_ip2 = CONFIG['actual_wear'].get(c_i+2, 0.0)   - theo_data['cumulative_vb'].get(f"cycle_{c_i+2}", 0.0)
        
        y_list.append([res_i, res_ip1, res_ip2])

        # 對應的 theo 值
        theo_steps_list.append([
            theo_data['cumulative_vb'].get(f"cycle_{c_i}", 0.0),
            theo_data['cumulative_vb'].get(f"cycle_{c_i+1}", 0.0),
            theo_data['cumulative_vb'].get(f"cycle_{c_i+2}", 0.0)
        ])

    x_all_np = np.array(x_list, dtype=np.float32)
    y_all_np = np.array(y_list, dtype=np.float32)
    theo_all_np = np.array(theo_steps_list, dtype=np.float32)

    # 轉成 Tensor
    x_all = torch.from_numpy(x_all_np).float()
    y_all = torch.from_numpy(y_all_np).float()
    theo_tensor = torch.from_numpy(theo_all_np).float()

    print(f"最終訓練樣本數: {len(x_all)} | 輸入維度: {x_all.shape[1]}")

    # ==================== 模型與訓練 ====================
    model = MLP(x_all.shape[1], hidden_dim=CONFIG['model']['hidden_dim'])
    model_path = os.path.join(CONFIG['paths']['model_dir'], 'model.pth')
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['model']['lr'])
    lambda_phys = CONFIG['model']['lambda_phys']

    print(f"模型參數量: {sum(p.numel() for p in model.parameters())}")

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(CONFIG['model']['epochs']):
        optimizer.zero_grad()

        pred_residual = model(x_all, theo_tensor)

        loss_data = nn.MSELoss()(pred_residual, y_all)
        loss_phys = nn.MSELoss()(pred_residual, torch.zeros_like(pred_residual))

        # 單調性約束
        final_pred_steps = theo_tensor + pred_residual
        loss_mono = torch.mean(torch.relu(final_pred_steps[:, 0] - final_pred_steps[:, 1])) + \
                    torch.mean(torch.relu(final_pred_steps[:, 1] - final_pred_steps[:, 2]))

        loss = (1 - lambda_phys) * loss_data + lambda_phys * loss_phys + 1.0 * loss_mono

        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch

        if epoch % 1000 == 0 or epoch == CONFIG['model']['epochs'] - 1:
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.2e} | "
                  f"Data: {loss_data.item():.2e} | Phys: {loss_phys.item():.2e} | "
                  f"Mono: {loss_mono.item():.2e}")

    print(f"Best Model (Loss = {best_loss:.2e}, Epoch {best_epoch})")

    # 儲存模型    
    torch.save(model.state_dict(), model_path)
    print(f"模型已儲存至: {model_path}")


if __name__ == "__main__":
    main('data_AB')