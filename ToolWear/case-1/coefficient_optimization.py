import os
import json
import torch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from theoretical_estimation import TheoreticalWearCalculator
from config import initialize_config
import time

start = time.time()


def main(path_key='data', target_cycles=None):    
    global CONFIG
    CONFIG = initialize_config(path_key)
    affix = CONFIG[path_key]['data_root'].lstrip('./\\')

    # 載入 processed data
    pkl_path = os.path.join(CONFIG['paths']['pkl_dir'], 
                           f"{CONFIG['paths']['processed_data']}_{affix}.pkl")
    with open(pkl_path, 'rb') as f:
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

    actual_dict = CONFIG['actual_wear']
    # initial_wear = CONFIG['wear_data']['machining_parameter'].get('initial_wear', 0.0)
    initial_wear = 0.0

    # 原始係數
    ref = CONFIG['theoretical_formula_coefficient']
    ref_G = ref['G_ABRASIVE']
    ref_A = ref['A_ADHESIVE']
    ref_B = ref['B_ADHESIVE']
    ref_E = ref['E_DIFFUSIVE']
    ref_R = ref['R_GAS']
    ref_poly = ref.get('diffusive_poly', {})

    print(f"初始磨耗: {initial_wear:.4f} mm | 目標 Cycle: {target_cycles}")

    # ================== 預處理（含 condition 資訊）==================
    def preprocess_all_cycles(processed):
        pre = {}
        for cycle_key, data in processed.items():
            cycle_num = int(cycle_key.split('_')[1])
            segments = []
            for seg in data.get('segments', []):
                seg_df = pd.DataFrame(seg['segment_df'])
                
                # BendingY / BendingX 自動切換
                if seg_df['BendingY'].std() < 0.3 or pd.isna(seg_df['BendingY'].std()):
                    max_force = float(seg_df['BendingX'].max())
                else:
                    max_force = float(seg_df['BendingY'].max())
                
                avg_temp = float(np.mean(seg_df.get('CutterDermisTemperature_C', 0)))
                duration = float(seg.get('duration', 1.0))
                
                segments.append({
                    'max_force': max_force,
                    'avg_temp': avg_temp,
                    'duration': duration
                })
            pre[cycle_num] = segments
        return pre

    preprocessed = preprocess_all_cycles(processed)

    # ================== Objective Function (跨條件版) ==================
    def objective(coeffs, preprocessed, actual_dict, initial_wear, target_set, start_cycle):
        G, A, B = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        loss = 0.0
        cum_wear = actual_dict.get(start_cycle - 1, initial_wear)

        for cycle_num in sorted(preprocessed.keys()):
            if cycle_num < start_cycle:
                cum_wear = actual_dict.get(cycle_num, cum_wear)
                continue

            # === 取得該 cycle 的加工條件 ===
            cond_info = CONFIG['wear_data']['indexed_cycles'].get(str(cycle_num), {})
            cond = cond_info.get("condition", "1")
            cycle_params = condition_params.get(cond, {})

            # 建立 calculator（每次都傳入對應 mach_param）
            calculator = TheoreticalWearCalculator(
                config=CONFIG,
                coefficients={
                    'G_ABRASIVE': G,
                    'A_ADHESIVE': A,
                    'B_ADHESIVE': B,
                    'E_DIFFUSIVE': ref_E,
                    'R_GAS': ref_R,
                    'diffusive_poly': ref_poly
                },
                machining_param=cycle_params
            )

            cycle_delta = 0.0
            for seg in preprocessed[cycle_num]:
                if seg['max_force'] <= 0:
                    continue
                Fn = calculator.calc_tool_normal_force(seg['max_force'])
                sigma_n = calculator.calc_orthogonal_stress(Fn)
                phi = calculator.calc_shear_angle()
                Vs = calculator.calc_sliding_velocity(phi)
                result = calculator.calc_total_wear_rate(sigma_n, Vs, seg['avg_temp'])
                cycle_delta += result.get('total_rate', 0.0) * seg['duration']

            cum_wear += cycle_delta

            if cycle_num in target_set:
                actual = actual_dict.get(cycle_num, 0.0)
                loss += (cum_wear - actual) ** 2

        return loss

    # ================== 優化設定 ==================
    bounds = [
        (ref_G * 0.01, ref_G * 100),
        (ref_A * 0.01, ref_A * 100),
        (ref_B * 0.01, ref_B * 100)   # B_ADHESIVE 通常變化較小
    ]

    initial_guess = np.array([ref_G, ref_A, ref_B], dtype=np.float64)
    target_set = set(target_cycles) if target_cycles else set()
    start_cycle = min(target_set) if target_set else 1

    print(f"開始 L-BFGS-B 優化 (從 Cycle {start_cycle} 開始)...")

    result = minimize(
        fun=objective,
        x0=initial_guess,
        args=(preprocessed, actual_dict, initial_wear, target_set, start_cycle),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 15000, 'eps': 1e-8, 'ftol': 1e-12}
    )

    fitted_G, fitted_A, fitted_B = result.x

    print("\n=== 優化完成 ===")
    print(f"Loss: {result.fun:.6e} | Success: {result.success}")
    print(f"G_ABRASIVE : {fitted_G:.6e} (x {fitted_G/ref_G:.3f})")
    print(f"A_ADHESIVE : {fitted_A:.6e} (x {fitted_A/ref_A:.3f})")
    print(f"B_ADHESIVE : {fitted_B:.4f} (x {fitted_B/ref_B:.3f})")

    # 儲存
    fitted_config = {
        "theoretical_formula_coefficient": {
            "G_ABRASIVE": float(f"{fitted_G:.6e}"),
            "A_ADHESIVE": float(f"{fitted_A:.6e}"),
            "B_ADHESIVE": float(f"{fitted_B:.4f}"),
            "E_DIFFUSIVE": ref_E,
            "R_GAS": ref_R,
            "diffusive_poly": ref_poly
        }
    }
    out_json = f'optimized_coefficients_{affix}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(fitted_config, f, indent=2)
    print(f"係數已儲存 → {out_json}")

    # ================== 完整驗證 ==================
    print("\n=== 新係數完整驗證 ===")
    cum_wear = initial_wear
    for cycle_num in sorted(preprocessed.keys()):
        cond_info = CONFIG['wear_data']['indexed_cycles'].get(str(cycle_num), {})
        cond = cond_info.get("condition", "1")
        cycle_params = condition_params.get(cond, {})

        calculator = TheoreticalWearCalculator(
            config=CONFIG,
            coefficients={
                'G_ABRASIVE': fitted_G, 'A_ADHESIVE': fitted_A, 'B_ADHESIVE': fitted_B,
                'E_DIFFUSIVE': ref_E, 'R_GAS': ref_R, 'diffusive_poly': ref_poly
            },
            machining_param=cycle_params
        )

        cycle_delta = 0.0
        for seg in preprocessed[cycle_num]:
            if seg['max_force'] <= 0: continue
            Fn = calculator.calc_tool_normal_force(seg['max_force'])
            sigma_n = calculator.calc_orthogonal_stress(Fn)
            phi = calculator.calc_shear_angle()
            Vs = calculator.calc_sliding_velocity(phi)
            res = calculator.calc_total_wear_rate(sigma_n, Vs, seg['avg_temp'])
            cycle_delta += res.get('total_rate', 0) * seg['duration']

        cum_wear += cycle_delta

        if target_cycles and cycle_num in target_cycles:
            actual = actual_dict.get(cycle_num, 0.0)
            print(f"Cycle {cycle_num:2d} [{cond}] | 預測: {cum_wear:.5f} | 實際: {actual:.5f} | 誤差: {cum_wear - actual:+.5f}")

    print(f"\n總耗時: {(time.time() - start):.2f} 秒")


if __name__ == "__main__":
    # main('data_C', target_cycles=[1, 73])
    main('data', target_cycles=[1, 36])