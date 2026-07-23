import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

os.chdir(os.path.dirname(__file__))

# ====================== 檔案路徑 ======================
file_paths = {
    "QIT-CEMC": "./QIT-CEMC/wear_data_QIT.json",
    "NKUST-A": "./NKUST-A/wear_data_A.json",
    "NKUST-B": "./NKUST-B/wear_data_B.json",
    "NKUST-C": "./NKUST-C/wear_data_C.json"
}

# ====================== 讀取資料 ======================
data = {}
for name, path in file_paths.items():
    with open(path, 'r', encoding='utf-8') as f:
        data[name] = json.load(f)

# ====================== 提取並計算加工參數 ======================
params = {}
for name, dataset in data.items():
    params[name] = {}
    
    if name in ["NKUST-A", "NKUST-B", "NKUST-C"]:
        # 這三個都有 conditions 結構
        params[name]["conditions"] = {}
        for cond_key, condition in dataset["conditions"].items():
            mp = condition["machining_parameter"]
            ap = mp["ap"]
            ae = mp["ae"]
            feed_rate = mp["feed_rate"]
            cutting_speed = mp["cutting_speed"]
            
            params[name]["conditions"][cond_key] = {
                "area": round(ap * ae, 4),
                "vc": round(cutting_speed, 1),
                "vf": round(feed_rate, 1),
                "ap": ap,
                "ae": ae,
            }
    else:
        # QIT-CEMC
        mp = dataset["machining_parameter"]
        ap = mp["ap"]
        ae = mp["ae"]
        feed_rate = mp["feed_rate"]
        cutting_speed = mp["cutting_speed"]
        
        params[name] = {
            "area": round(ap * ae, 4),
            "vc": round(cutting_speed, 1),
            "vf": round(feed_rate, 1),
            "ap": ap,
            "ae": ae,
        }

print("提取與計算的加工參數：")
for name, p in params.items():
    if "conditions" in p:
        print(f"\n{name}:")
        for cond, val in p["conditions"].items():
            print(f"  Cond{cond}: ap×ae={val['area']} mm², "
                  f"Vc={val['vc']} m/min, "
                  f"Vf={val['vf']} mm/min")
    else:
        print(f"{name}: ap×ae={p['area']} mm², "
              f"Vc={p['vc']} m/min, "
              f"Vf={p['vf']} mm/min")

# ====================== 繪製 3D 圖 ======================
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')

colors = {'QIT-CEMC': 'blue', 'NKUST-A': 'orange', 'NKUST-B': 'green', 'NKUST-C': 'red'}
markers = {'QIT-CEMC': '^', 'NKUST-A': 'o', 'NKUST-B': 'o', 'NKUST-C': 'o'}

# 用來確保圖例只出現一次
legend_handles = {}

# 繪製散點
for name, p in params.items():
    if "conditions" in p:
        for cond_name, cond in p["conditions"].items():
            handle = ax.scatter(cond["area"], cond["vc"], cond["vf"],
                                color=colors[name], marker=markers[name], s=100)
            if name not in legend_handles:
                legend_handles[name] = handle
    else:
        handle = ax.scatter(p["area"], p["vc"], p["vf"],
                            color=colors[name], marker=markers[name], s=100)
        legend_handles[name] = handle

# ====================== 連線 ======================
# NKUST-A Cond1 → Cond2
if "NKUST-A" in params:
    a = params["NKUST-A"]["conditions"]
    ax.plot([a["1"]["area"], a["2"]["area"]],
            [a["1"]["vc"],   a["2"]["vc"]],
            [a["1"]["vf"],   a["2"]["vf"]],
            color='orange', linewidth=2.5, linestyle='-')

# NKUST-B Cond1 → Cond2
if "NKUST-B" in params:
    b = params["NKUST-B"]["conditions"]
    ax.plot([b["1"]["area"], b["2"]["area"]],
            [b["1"]["vc"],   b["2"]["vc"]],
            [b["1"]["vf"],   b["2"]["vf"]],
            color='green', linewidth=2.5, linestyle='-')

# NKUST-C Cond1 → Cond2
# if "NKUST-C" in params:
#     c = params["NKUST-C"]["conditions"]
#     ax.plot([c["1"]["area"], c["2"]["area"]],
#             [c["1"]["vc"],   c["2"]["vc"]],
#             [c["1"]["vf"],   c["2"]["vf"]],
#             color='red', linewidth=2.5, linestyle='-')

# ====================== NKUST-C 三角形面 ======================
if "NKUST-C" in params:
    c = params["NKUST-C"]["conditions"]
    # 使用 NKUST-C 的三個點組成三角形（C1, C2, 以及一個組合點）
    points_C = [
        (c["1"]["area"], c["1"]["vc"], c["1"]["vf"]),
        (c["2"]["area"], c["2"]["vc"], c["2"]["vf"]),
        (c["3"]["area"], c["3"]["vc"], c["3"]["vf"])   # 組合點形成三角形
    ]
    poly_C = Poly3DCollection([points_C], alpha=0.3, facecolor='red', edgecolor='red', linewidth=2.5)
    ax.add_collection3d(poly_C)

ax.legend(legend_handles.values(), legend_handles.keys(), fontsize=14, loc='upper left')

# 軸標籤與標題
ax.set_xlabel('Area (Ap × Ae) (mm²)', fontsize=16, labelpad=10)
ax.set_ylabel('Cutting Speed Vc (m/min)', fontsize=16, labelpad=10)
ax.set_zlabel('Feed Rate Vf (mm/min)', fontsize=16, labelpad=10)
ax.set_title('Comparison of 3D Machining Parameters', fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)

ax.view_init(elev=15, azim=-110)

# 儲存與顯示
save_path = './comparison_3d_machining_parameters.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n3D 圖已儲存至：{save_path}")