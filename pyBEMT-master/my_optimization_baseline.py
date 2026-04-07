from pybemt.solver import Solver
import numpy as np

# 1. 加载配置（假设使用示例文件）
solver = Solver('examples/true_baseline.ini')

# 2. 设置初始翼型参数（权重全部设为 0）
num_sections = len(solver.rotor.sections)
baseline_weights = [0.0] * 18 # 18个0，代表原始 Clark Y 翼型

for i in range(num_sections):
    solver.rotor.sections[i].airfoil.weights = baseline_weights

# 3. 运行求解器
print("开始运行 AI 驱动的 BEMT 求解...")
# run() 函数内部会打印 Trust, Torque, Power
solver.run() 

# 4. 提取物理量并计算
T = solver.T        # 推力 (N)
Q = solver.Q        # 扭矩 (Nm)
P = solver.P        # 功率 (W)
rho = solver.fluid.rho
n = solver.rpm / 60.0
D = solver.rotor.diameter
V = solver.v_inf

# 计算无量纲系数
CT = T / (rho * (n**2) * (D**4))
CP = P / (rho * (n**3) * (D**5))
eta = (T * V) / P if P > 0 else 0.0

print("\n" + "="*45)
print("      CLARK Y 基准性能评估报告 (Baseline)      ")
print("="*45)
print(f"推力 (Thrust):  {T:.4f} N")
print(f"扭矩 (Torque):  {Q:.4f} Nm")
print(f"功率 (Power):   {P:.4f} W")
print("-" * 45)
print(f"推力系数 (CT):  {CT:.6f}")
print(f"功率系数 (CP):  {CP:.6f}")
print(f"螺旋桨效率 (η): {eta*100:.2f} %")
print("="*45)