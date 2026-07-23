import os
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from config import initialize_config, add_condition_background

start = time.time()


class TheoreticalWearCalculator:
    """
    使用正確公式計算理論磨耗
    支援從 config.json 讀取多組係數，並支援不同加工條件
    """
    def __init__(self, config=None, coefficients: dict = None, machining_param: dict = None):
        self.config = config if config is not None else CONFIG
        self.load_from_config()
        if coefficients is not None:
            self.set_coefficients(coefficients)
        
        tool_parameter = self.config['wear_data']['tool']
        self.moment_arm = tool_parameter['straingauge_to_nose_length'] + tool_parameter['tool_overhang_length']
        # 使用該 cycle 對應的加工參數
        # self.machining_parameter = machining_param or {}
        self.machining_parameter = machining_param or self.config['wear_data'].get('machining_parameter', {})
        # self.machining_parameter = self.config['wear_data']['machining_parameter']
        
        # self.Vc = tool_parameter['diameter'] * machining_parameter['spindle_speed'] * math.pi / 60
        self.Vc = self.machining_parameter['cutting_speed'] * 1000 / 60
        self.alpha_t = tool_parameter['rake_angle']
        self.ap = self.machining_parameter['ap']
        self.ae = self.machining_parameter['ae']
        self.initial_wear = self.machining_parameter['initial_wear']
        self.chip_ratio = self.machining_parameter['ap'] / self.machining_parameter['chip_thickness']
        self.contact_area = self.ae * self.ap

    def load_from_config(self):
        """從 config 載入預設係數"""
        theo_cfg = self.config['theoretical_formula_coefficient']
        self._set_coefficients_from_dict(theo_cfg)

    def set_coefficients(self, coeffs: dict):
        """動態設定係數"""
        self._set_coefficients_from_dict(coeffs)

    def _set_coefficients_from_dict(self, cfg: dict):
        self.G_ABRASIVE = cfg['G_ABRASIVE']
        self.A_ADHESIVE = cfg['A_ADHESIVE']
        self.B_ADHESIVE = cfg['B_ADHESIVE']
        self.E_DIFFUSIVE = cfg['E_DIFFUSIVE']
        self.R_GAS = cfg['R_GAS']
        
        poly = cfg['diffusive_poly']
        self.a = poly['a']
        self.b = poly['b']
        self.c = poly['c']
        self.d = poly['d']

    def calc_tool_normal_force(self, force: float) -> float:
        return force / 1000 / self.moment_arm
    
    def calc_orthogonal_stress(self, Fn: float) -> float:
        return Fn / self.contact_area

    def calc_shear_angle(self) -> float:
        r = self.chip_ratio
        alpha_rad = np.deg2rad(self.alpha_t)
        tan_phi = (r * np.cos(alpha_rad)) / (1 - r * np.sin(alpha_rad))
        phi = np.rad2deg(np.arctan(tan_phi))
        return phi
    
    def calc_sliding_velocity(self, phi: float) -> float:
        phi_rad = np.deg2rad(phi)
        alpha_rad = np.deg2rad(self.alpha_t)
        Vs = self.Vc * np.sin(phi_rad) / np.cos(phi_rad - alpha_rad)
        return Vs
    
    def calc_abrasive_wear_rate(self) -> float:
        return self.G_ABRASIVE * self.Vc
    
    def calc_adhesive_wear_rate(self, Vs: float, sigma_n: float, Tm_K: float) -> float:
        exponent = -self.B_ADHESIVE / Tm_K
        return self.A_ADHESIVE * sigma_n * Vs * np.exp(exponent)
    
    def calc_diffusive_coefficient(self, Tm_K: float) -> float:
        return self.a * Tm_K**3 + self.b * Tm_K**2 + self.c * Tm_K + self.d
    
    def calc_diffusive_wear_rate(self, Tm_K: float) -> float:
        D = self.calc_diffusive_coefficient(Tm_K)
        exponent = -self.E_DIFFUSIVE / (self.R_GAS * Tm_K)
        return D * np.exp(exponent)
    
    def calc_total_wear_rate(self, sigma_n: float, Vs: float, Tm_C: float):
        Tm_K = Tm_C + 273.15
        abrasive = self.calc_abrasive_wear_rate()
        adhesive = self.calc_adhesive_wear_rate(Vs, sigma_n, Tm_K)
        diffusive = self.calc_diffusive_wear_rate(Tm_K)
        
        if Tm_K > 973.15:
            total = adhesive + diffusive
        else:
            total = abrasive + adhesive
        return {"total_rate": total}


def main(path_key='data', switch_cycles: list = None):
    global CONFIG
    CONFIG = initialize_config(path_key)
    # suffix = CONFIG[path_key]['data_root'].split('-')[-1]
    affix = CONFIG[path_key]['data_root'].lstrip('./\\')

    # 載入 processed data
    with open(os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['processed_data']}_{affix}.pkl"), 'rb') as f:
        processed = torch.load(f)
    print(f"資料載入完成，耗時 {(time.time() - start):.2f} 秒")

    # ==================== 支援單一條件 ====================
    conditions = CONFIG['wear_data'].get('conditions', {})
    default_mach_param = CONFIG['wear_data'].get('machining_parameter', {})
    
    if conditions:
        condition_params = {cond: data.get('machining_parameter', {}) 
                           for cond, data in conditions.items()}
        print("使用多條件模式")
    else:
        condition_params = {"1": default_mach_param}
        print("使用單一條件模式")

    # === 從 config.json 讀取多組係數 ===
    theo_coefficients = []
    i = 1
    while True:
        key = "theoretical_formula_coefficient" if i == 1 else f"theoretical_formula_coefficient_{i}"
        if key in CONFIG:
            theo_coefficients.append(CONFIG[key])
        else:
            break
        i += 1


    # === 處理 switch_cycle ===
    switch_cycles_sorted = sorted(switch_cycles) if switch_cycles else []
    print(f"係數切換點：{switch_cycles} | 共使用 {len(theo_coefficients)} 組係數")

    theo_wear = {}
    theo_rate_dict = {}
    cycle_nums = []
    theo_values = []
    actual_values = []
    cumulative_wear = 0.0

    # calculator = TheoreticalWearCalculator(coefficients=theo_coefficients[0])
    # cumulative_wear = calculator.initial_wear
    # switch_cycles_sorted = sorted(switch_cycles)

    print("開始計算理論磨耗（支援混合條件）")

    for cycle, data in processed.items():
        cycle_num = int(cycle.split('_')[1])
        
        # 透過 condition 取得該 cycle 的加工參數
        cond_info = CONFIG['wear_data']['indexed_cycles'].get(str(cycle_num), {})
        cond = cond_info.get("condition", "1")
        cycle_params = condition_params.get(cond, {})

        # 決定使用哪一組係數
        combo_idx = 0
        if switch_cycles_sorted:  # 只有有切換點時才計算
            for idx, sp in enumerate(switch_cycles_sorted):
                if cycle_num > sp:
                    combo_idx = idx + 1
                else:
                    break
        
        calculator = TheoreticalWearCalculator(
            config=CONFIG,
            coefficients=theo_coefficients[combo_idx],
            machining_param=cycle_params
        )

        segments = data.get('segments', [])
        cycle_delta_wear = 0.0
        cycle_wear_rates = []

        for seg in segments:
            seg_df = pd.DataFrame(seg['segment_df'])

            # 判斷 Y 通道是否為有效訊號，如果沒接應變規，標準差會非常接近 0
            if seg_df['BendingY'].std() < 0.3 or pd.isna(seg_df['BendingY'].std()):
                bending = seg_df['BendingX'].max() # Y 通道判斷為雜訊，切換至 X 方向
                # print(f"偵測到 BendingY 為雜訊(std={seg_df['BendingY'].std():.4f})，已自動切換至 BendingX")
            else: bending = seg_df['BendingY'].max()

            Fn = calculator.calc_tool_normal_force(bending)
            avg_temp = np.mean(seg_df['CutterDermisTemperature_C'])

            sigma_n = calculator.calc_orthogonal_stress(Fn)
            phi = calculator.calc_shear_angle()
            Vs = calculator.calc_sliding_velocity(phi)
            
            result = calculator.calc_total_wear_rate(sigma_n, Vs, avg_temp)
            wear_rate = result['total_rate']
            
            theo_vb_this = wear_rate * seg['duration']
            cycle_delta_wear += theo_vb_this
            cycle_wear_rates.append(wear_rate)

        cumulative_wear += cycle_delta_wear # 更新累積磨耗
        
        theo_wear[cycle] = float(cumulative_wear)
        theo_rate_dict[cycle] = float(np.mean(cycle_wear_rates)) if cycle_wear_rates else 0.0

        cycle_nums.append(cycle_num)
        theo_values.append(cumulative_wear)
        actual_values.append(CONFIG['actual_wear'].get(cycle_num, 0.0))

        print(f"Cycle {cycle_num:2d} | Condition: {cond} | "
              f"Duration: {data.get('total_machining_duration', 0):.2f}s | "
              f"Wear Rate: {theo_rate_dict[cycle]:.8f} | "
              f"Cumulative: {cumulative_wear:.5f} mm | Combo: {combo_idx+1}")

    # 儲存結果
    theo_data = {"cumulative_vb": theo_wear, "wear_rate": theo_rate_dict}
    with open(os.path.join(CONFIG['paths']['pkl_dir'], f"{CONFIG['paths']['theoretical_estimation']}_{affix}.pkl"), 'wb') as f:
        torch.save(theo_data, f)

    # 評估指標
    mae = mean_absolute_error(actual_values, theo_values)
    mape = mean_absolute_percentage_error(actual_values, theo_values) * 100
    mse = mean_squared_error(actual_values, theo_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, theo_values)
    print(f"\nMAE  = {mae:.4f} mm")
    print(f"MAPE = {mape:.2f} %")
    print(f"RMSE = {rmse:.4f} mm")
    print(f"R²   = {r2:.4f}")

    # 視覺化
    plt.figure(figsize=(18, 8))

    # === 加入條件背景色塊 ===
    ax = plt.gca()
    
    plt.plot(cycle_nums, theo_values, 'o-', color='green', linewidth=2, label='Theoretical')
    plt.plot(cycle_nums, actual_values, 's--', color='blue', linewidth=2, label='Actual')
    
    # if switch_cycles_sorted:
    #     for sp in switch_cycles_sorted:
    #         plt.axvline(x=sp, color='yellow', linestyle='--', alpha=0.5)
    
    plt.xlabel('Machining Cycle', fontsize=20)
    plt.ylabel('Tool Wear VB (mm)', fontsize=20)
    plt.title(f'{affix} Theoretical VS Actual Tool Wear', fontsize=24, fontweight='bold')

    # 設定好 Y 軸極限後再畫背景
    ax.set_ylim(bottom=0)
    # === 加入條件背景色塊 ===
    bg_patches = add_condition_background(ax)

    # === 圖例處理 ===
    legend_handles = [
        Line2D([0], [0], color='green', marker='o', linestyle='-', linewidth=2, label='Theoretical'),
        Line2D([0], [0], color='blue', marker='s', linestyle='--', linewidth=2, label='Actual')
    ]
    # legend_handles.extend(bg_patches)
    plt.legend(handles=legend_handles, fontsize=14, loc='upper left')

    # plt.legend(fontsize=12)
    plt.grid(True, which='major', linestyle='-', alpha=0.7)    
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5)) # 平均切成 5 等份
    plt.grid(True, which='minor', linestyle=':', alpha=0.7) # 次要格線
    plt.xticks(range(min(cycle_nums), max(cycle_nums)+1, 2))
    plt.tick_params(axis='both', which='major', labelsize=16)

    ax.set_xlim(left=min(cycle_nums) - 1, right=max(cycle_nums) + 1)
    ax.set_ylim(bottom=0)

    viz_path = os.path.join(CONFIG['paths']['visualization_dir'], f'{affix}_theoretical_estimation.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"圖表已儲存 → {viz_path}")
    print(f"總耗時: {(time.time() - start):.2f} 秒")
    plt.show()

if __name__ == "__main__":
    main('data_C', switch_cycles=[])   # A 到 B 的切換點建議設在 cycle 18
    # main('data', switch_cycles=[])  # 空列表 → 不切換
    # main('data', switch_cycles=[10])           # 單次切換
    # main('data', switch_cycles=[19, 25])       # 兩次切換