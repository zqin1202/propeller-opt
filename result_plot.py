import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 将包含 hicks_henne_deform 的文件夹加入路径
# 假设你在 pyBEMT-master 目录下运行，hicks_henne_deform 在 ../xfoil_runner/ 里
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xfoil_runner'))

try:
    from hicks_henne_deform import apply_hicks_henne
except ImportError:
    print("找不到 hicks_henne_deform 模块，请确保相对路径正确！")
    sys.exit()

if __name__ == "__main__":
    # 指向你的基础翼型文件 (请根据你电脑里的实际位置修改，这里假设在上一级的 xfoil_runner 里)
    base_file = "../xfoil_runner/CLARKY_geo.dat" 
    
    # 填入 PSO 刚刚跑出来的最佳 18 维参数
    best_weights = [
         0.01 ,  -0.01 ,  -0.01 ,  -0.01 ,  -0.01  , -0.01  , -0.0064 ,-0.0068 , 0.0003,
 -0.0038, -0.0071,  0.   ,   0.005 , -0.0097, -0.0009 ,-0.  ,   -0.01  ,  0.01
    ]
    
    print("正在生成最佳水空两栖优化翼型对比图...")
    try:
        x_new, y_new, x_std, y_up_base, y_low_base = apply_hicks_henne(base_file, best_weights, "Optimized_Airfoil.dat")
        
        # 开始绘图
        plt.figure(figsize=(12, 5), dpi=150)
        
        # 画出基础的 CLARK Y 翼型 (灰色虚线)
        x_base_plot = np.concatenate((x_std[::-1], x_std[1:]))
        y_base_plot = np.concatenate((y_up_base[::-1], y_low_base[1:]))
        plt.plot(x_base_plot, y_base_plot, color='gray', linestyle='--', linewidth=2, label='Baseline (CLARK Y)')
        
        # 画出优化后的新翼型 (红色实线)
        plt.plot(x_new, y_new, color='#d62728', linewidth=2.5, label='Optimized Amphibious Airfoil')
        
        # 填充颜色让对比更明显
        plt.fill(x_base_plot, y_base_plot, color='gray', alpha=0.1)
        plt.fill(x_new, y_new, color='#d62728', alpha=0.15)
        
        plt.title("Baseline vs. Optimized Airfoil Shape (Weighted 50% Air / 50% Water)", fontsize=14, fontweight='bold')
        plt.xlabel("Chord Fraction (x/c)", fontsize=12)
        plt.ylabel("Thickness Fraction (y/c)", fontsize=12)
        plt.axis('equal') 
        plt.legend(fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.8)
        
        # 保存并显示
        plt.savefig("Airfoil_Comparison.png", bbox_inches='tight')
        print("✅ 绘图成功！图片已保存为 Airfoil_Comparison.png，同时弹出显示窗口。")
        plt.show()
        
    except FileNotFoundError:
        print(f"找不到基础翼型文件: {base_file}。请检查路径！")