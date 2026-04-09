import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# ================= 核心：Hicks-Henne 变形函数 =================
def apply_hicks_henne(x_base, y_base, alphas, is_upper):
    """应用 Hicks-Henne 型函数进行变形"""
    # 预设的9个控制点位置 (参考常见设定，需与你生成数据库时保持一致)
    x_c = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9])
    widths = np.ones(9) * 0.85
    
    y_new = np.copy(y_base)
    for i in range(9):
        # 确保 x_base 在 0~1 之间，避免 log(0) 警告
        x_safe = np.clip(x_base, 1e-6, 1.0)
        bump = np.sin(np.pi * x_safe ** (np.log(0.5) / np.log(x_c[i]))) ** 2
        amplitude = alphas[i]
        if is_upper:
            y_new += amplitude * bump
        else:
            y_new += amplitude * bump
    return y_new

def generate_and_test_airfoil():
    print("="*50)
    print(" 翼型几何排错与 XFOIL 测试台 ")
    print("="*50)
    
    # 1. 确保基础文件存在
    base_file = "CLARKY_geo.dat" # 确保这里是几何文件！
    if not os.path.exists(base_file):
        print(f"致命错误：找不到基础几何文件 {base_file}")
        return

    # 2. 读取基础几何并拆分上下表面 (假设标准 Selig 格式: 后缘->前缘->后缘)
    coords = np.loadtxt(base_file, skiprows=1)
    le_idx = np.argmin(coords[:, 0])
    x_up, y_up = coords[:le_idx+1, 0][::-1], coords[:le_idx+1, 1][::-1]
    x_low, y_low = coords[le_idx:, 0], coords[le_idx:, 1]

    # 统一插值到标准化 x 网格上 (使用余弦分布加密前尾缘)
    beta = np.linspace(0, np.pi, 100)
    x_std = 0.5 * (1 - np.cos(beta))
    
    y_up_std = np.interp(x_std, x_up, y_up)
    y_low_std = np.interp(x_std, x_low, y_low)

    # 3. 接收用户输入的 18 个参数
    print("\n请输入18个 Hicks-Henne 控制参数 (用空格分隔，例如: 0.01 -0.005 ...)：")
    try:
        user_input = input(">> ")
        alphas = np.array([float(x) for x in user_input.split()])
        if len(alphas) != 18:
            raise ValueError(f"参数数量错误！需要18个，你输入了 {len(alphas)} 个。")
    except Exception as e:
        print(f"输入解析失败: {e}")
        return

    alphas_up = alphas[:9]
    alphas_low = alphas[9:]

    # 4. 生成变形后的上下表面
    y_up_def = apply_hicks_henne(x_std, y_up_std, alphas_up, is_upper=True)
    y_low_def = apply_hicks_henne(x_std, y_low_std, alphas_low, is_upper=False)

    # 组装为完整的翼型坐标写入文件
    x_full = np.concatenate([x_std[::-1], x_std[1:]])
    y_full = np.concatenate([y_up_def[::-1], y_low_def[1:]])
    
    deformed_file = "test_deformed.dat"
    with open(deformed_file, 'w') as f:
        f.write("TEST_AIRFOIL\n")
        for x, y in zip(x_full, y_full):
            f.write(f"{x:.6f} {y:.6f}\n")

    # 5. 几何图形强制核查 (这一步是你发现“尾缘极薄/交叉”的核心)
    plt.figure(figsize=(10, 4))
    plt.plot(x_std, y_up_std, 'k--', label='Base Upper', alpha=0.5)
    plt.plot(x_std, y_low_std, 'k--', label='Base Lower', alpha=0.5)
    plt.plot(x_std, y_up_def, 'r-', label='Deformed Upper', linewidth=2)
    plt.plot(x_std, y_low_def, 'b-', label='Deformed Lower', linewidth=2)
    
    # 突出显示可能交叉的区域
    thickness = y_up_def - y_low_def
    invalid_idx = np.where(thickness <= 0)[0]
    if len(invalid_idx) > 0:
        print("\n⚠️ 严重警告：发现上下表面交叉或重合！")
        plt.scatter(x_std[invalid_idx], y_up_def[invalid_idx], color='magenta', s=50, zorder=5, label='Intersection/Invalid')

    plt.title("Airfoil Geometry Inspection")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 6. 调用 XFOIL 进行气动测试
    Re = float(input("\n请输入测试雷诺数 Re (例如 500000): "))
    AoA = float(input("请输入测试迎角 Alpha (例如 4.0): "))
    
    print("\n正在启动 XFOIL...")
    xfoil_script = f"""
    load {deformed_file}
    pane
    oper
    visc {Re}
    iter 100
    pacc
    test_polar.txt

    alfa {AoA}
    pacc
    quit
    """
    
    # 清理旧的极线文件
    if os.path.exists("test_polar.txt"):
        os.remove("test_polar.txt")

    try:
        # 调用 XFOIL (假设当前目录下有 xfoil.exe)
        process = subprocess.Popen("xfoil.exe", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = process.communicate(xfoil_script)
        
        # 解析极线结果
        if os.path.exists("test_polar.txt"):
            with open("test_polar.txt", 'r') as f:
                lines = f.readlines()
                # 寻找包含数据的行 (通常跳过表头)
                for line in lines:
                    parts = line.split()
                    # 粗略判断是不是数据行 (包含多个浮点数)
                    if len(parts) >= 3 and parts[0].replace('.','',1).replace('-','',1).isdigit():
                        print("\n>>> XFOIL 计算成功！ <<<")
                        print(f"Alpha: {parts[0]}, Cl: {parts[1]}, Cd: {parts[2]}")
                        return
            print("\n❌ XFOIL 运行结束，但在极线文件中未找到有效收敛数据。可能是翼型过于畸形导致不收敛。")
        else:
            print("\n❌ XFOIL 未生成极线文件。")
    except Exception as e:
        print(f"执行 XFOIL 时出错: {e}")

if __name__ == "__main__":
    generate_and_test_airfoil()