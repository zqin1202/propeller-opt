import numpy as np
from pybemt.solver import Solver
import os

# ==========================================
# 1. 录入你提取的 Table 2 原始数据
# ==========================================
r_raw = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12])
c_raw = np.array([0.0135, 0.016, 0.019, 0.0225, 0.024, 0.025, 0.025, 0.023, 0.0205, 0.017, 0.013, 0.0095])
theta_raw = np.array([52, 35, 28, 22, 20, 18, 16, 14, 13, 12, 10, 9])

# ==========================================
# 2. 物理逻辑修正：截断并对齐轮毂边缘 (R_hub = 0.012m)
# ==========================================
# 我们生成一个从 0.012 开始的真实叶片半径数组
r_blade = np.array([0.012, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12])

# 使用线性插值，求出 r=0.012 处的弦长和扭转角，其余点保持原样
c_blade = np.interp(r_blade, r_raw, c_raw)
theta_blade = np.interp(r_blade, r_raw, theta_raw)

# ==========================================
# 3. 动态生成 pyBEMT 配置文件 (true_baseline.ini)
# ==========================================
# 将 numpy 数组转为空格分隔的字符串
r_str = " ".join([f"{x:.4f}" for x in r_blade])
c_str = " ".join([f"{x:.4f}" for x in c_blade])
p_str = " ".join([f"{x:.2f}" for x in theta_blade])
sec_str = " ".join(["CLARKY"] * len(r_blade))

ini_content = f"""[case]
v_inf = 2.0
rpm = 7200

[fluid]
rho = 1.225
mu = 1.81e-5

[rotor]
nblades = 3
diameter = 0.24
radius_hub = 0.012

radius = {r_str}
section = {sec_str}
chord = {c_str}
pitch = {p_str}
"""

with open('true_baseline.ini', 'w') as f:
    f.write(ini_content)

print("已根据 Table 1 & Table 2 成功生成配置文件: true_baseline.ini")

# ==========================================
# 4. 运行求解器并输出结果
# ==========================================
print("正在调用 BEMT 求解器进行计算...\n")
try:
    my_solver = Solver('true_baseline.ini')
    T, Q, P, df_sections = my_solver.run()
    J, CT, CQ, CP, eta = my_solver.rotor_coeffs(T, Q, P)

    print("="*45)
    print("【论文真实基线模型 (True Baseline) 测算结果】")
    print("-" * 45)
    print(f"实际测算总推力 (T)    : {T:.4f} N")
    print(f"原论文目标推力 (Target): 6.86 N")
    
    # 计算相对误差
    error = abs(T - 6.86) / 6.86 * 100
    print(f"推力复刻误差          : {error:.2f} %")
    print("-" * 45)
    print(f"总消耗扭矩 (Q)        : {Q:.4f} N·m")
    print(f"总消耗功率 (P)        : {P:.2f} W")
    print(f"综合推进效率 (η)      : {eta * 100:.2f} %")
    print("="*45)
    
except Exception as e:
    print(f"计算失败，请检查 CLARKY.dat 文件是否在正确的 airfoils 目录下！报错信息：{e}")