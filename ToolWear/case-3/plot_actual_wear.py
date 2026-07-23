import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from config import initialize_config, add_condition_background

def plot_actual_wear(data):
    """只繪製實際磨耗圖"""
    
    # 初始化 CONFIG
    CONFIG = initialize_config(data)
    affix = CONFIG[data]['data_root'].lstrip('./\\')

    actual_wear = CONFIG['actual_wear']
    cycle_nums = sorted(actual_wear.keys())
    values = [actual_wear[c] for c in cycle_nums]

    plt.figure(figsize=(18, 8))
    ax = plt.gca()

    plt.plot(cycle_nums, values, 's--', color='blue', linewidth=2, label='Actual')
    plt.xlabel('Machining Cycle', fontsize=20)
    plt.ylabel('Tool Wear VB (mm)', fontsize=20)
    plt.title(f'{affix} Actual Tool Wear', fontsize=24, fontweight='bold')

    # 設定好 Y 軸極限後再畫背景
    ax.set_ylim(bottom=0)
    # === 加入條件背景色塊 ===
    bg_patches = add_condition_background(ax)

    legend_handles = [
        Line2D([0], [0], color='blue', marker='s', linestyle='--', linewidth=2, label='Actual')
    ]

    plt.legend(handles=legend_handles, fontsize=14, loc='upper left')
    plt.grid(True, which='major', linestyle='-', alpha=0.7)    
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5)) # 平均切成 5 等份
    plt.grid(True, which='minor', linestyle=':', alpha=0.7) # 次要格線
    plt.xticks(range(min(cycle_nums), max(cycle_nums)+1, 2))
    plt.tick_params(axis='both', which='major', labelsize=16)

    ax.set_xlim(left=min(cycle_nums) - 1, right=max(cycle_nums) + 1)
    ax.set_ylim(bottom=0)
    # 儲存圖片
    save_path = os.path.join(CONFIG['paths']['visualization_dir'], f'{affix}_actual_wear.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_actual_wear('data_C')