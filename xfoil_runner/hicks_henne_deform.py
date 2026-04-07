import numpy as np
import matplotlib.pyplot as plt

def apply_hicks_henne(base_dat_path, weights, output_path="deformed_airfoil.dat"):
    """
    读取基础翼型，施加 18个 Hicks-Henne 参数，并导出新翼型坐标
    :param base_dat_path: 真正的基础翼型坐标文件路径 (如 'CLARKY.dat')
    :param weights: 包含 18 个权重的数组 (前 9个控制上表面，后 9个控制下表面)
    :param output_path: 导出的新畸形翼型文件名
    """
    # ==========================================
    # 1. 强健地读取基础翼型坐标文件
    # ==========================================
    coords = []
    with open(base_dat_path, 'r') as f:
        for line in f:
            parts = line.split()
            # 只提取正好有两列数字的行，自动跳过表头文字
            if len(parts) == 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
                    
    coords = np.array(coords)
    x_base = coords[:, 0]
    y_base = coords[:, 1]

    # ==========================================
    # 2. 寻找前缘点 (Leading Edge)，切分上下表面
    # ==========================================
    # x 最小的那个点就是前缘点 (0,0) 附近
    le_index = np.argmin(x_base)
    
    # 提取上表面 (从前缘到后缘，确保 x 是从小到大 0 -> 1)
    x_upper_raw = x_base[:le_index+1][::-1] 
    y_upper_raw = y_base[:le_index+1][::-1]
    
    # 提取下表面 (从前缘到后缘，x 也是 0 -> 1)
    x_lower_raw = x_base[le_index:]
    y_lower_raw = y_base[le_index:]

    # ==========================================
    # 3. 重新网格化 (Re-paneling) - 极其重要！
    # ==========================================
    # 使用半余弦分布生成 100 个标准的 x 坐标点，确保两端密、中间疏
    beta = np.linspace(0, np.pi, 100)
    x_std = 0.5 * (1 - np.cos(beta))

    # 使用线性插值，将原始上下表面的 y 值映射到这 100 个标准 x 点上
    y_upper_std = np.interp(x_std, x_upper_raw, y_upper_raw)
    y_lower_std = np.interp(x_std, x_lower_raw, y_lower_raw)

    # ==========================================
    # 4. 施加 Hicks-Henne 凸块函数变形
    # ==========================================
    # 9 个凸块的中心位置
    peak_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    y_upper_new = np.copy(y_upper_std)
    y_lower_new = np.copy(y_lower_std)

    for i in range(9):
        # 计算形状控制系数 m_i
        m_i = np.log(0.5) / np.log(peak_x[i])
        # 计算凸块函数
        bump = (np.sin(np.pi * x_std**m_i))**3
        
        # 叠加权重
        y_upper_new += weights[i] * bump       # w_1 ~ w_9 控制上表面
        y_lower_new += weights[i+9] * bump     # w_10 ~ w_18 控制下表面

    # ==========================================
    # 5. 重新组装成 XFOIL 标准格式 (后缘 -> 上表面 -> 前缘 -> 下表面 -> 后缘)
    # ==========================================
    x_final = np.concatenate((x_std[::-1], x_std[1:]))
    y_final = np.concatenate((y_upper_new[::-1], y_lower_new[1:]))

    # ==========================================
    # 6. 保存为新的 .dat 文件
    # ==========================================
    with open(output_path, 'w') as f:
        f.write(f"Hicks-Henne_Deformed_Airfoil\n")
        for i in range(len(x_final)):
            f.write(f"{x_final[i]:.6f}  {y_final[i]:.6f}\n")

    return x_final, y_final, x_std, y_upper_std, y_lower_std


# ==========================================
# 测试区：一键直观对比变形前后的区别
# ==========================================
if __name__ == "__main__":
    # 确保你的文件夹里有真正的 CLARKY.dat
    base_file = "CLARKY_geo.dat" 
    
    # 模拟 PSO 算法随机生成的 18 个权重变量 (范围设定在 -0.015 到 +0.015 之间)
    np.random.seed(42)
    random_weights = np.random.uniform(-0.015, 0.015, 18)
    
    # 调用函数生成畸形翼型
    try:
        x_new, y_new, x_std, y_up_base, y_low_base = apply_hicks_henne(base_file, random_weights)
        print("成功生成 deformed_airfoil.dat 坐标文件！")
        
        # 画图对比
        plt.figure(figsize=(10, 4), dpi=120)
        
        # 画出基础的 CLARK Y 翼型 (灰色虚线)
        x_base_plot = np.concatenate((x_std[::-1], x_std[1:]))
        y_base_plot = np.concatenate((y_up_base[::-1], y_low_base[1:]))
        plt.plot(x_base_plot, y_base_plot, color='gray', linestyle='--', linewidth=1.5, label='Baseline (CLARK Y)')
        
        # 画出被 18 个参数改变后的新翼型 (红色实线)
        plt.plot(x_new, y_new, color='#d62728', linewidth=2, label='Deformed Airfoil (18 HH params)')
        
        plt.title("Airfoil Parameterization using Hicks-Henne Functions")
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.axis('equal') 
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()
        
    except FileNotFoundError:
        print(f"找不到 {base_file}！请务必找到真正的纯坐标 .dat 文件放到当前目录下。")