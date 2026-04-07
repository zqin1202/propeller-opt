import pandas as pd
from pybemt.solver import Solver

def evaluate_baseline():
    print("正在加载 Table 1 设定的基线螺旋桨参数...")
    
    # 实例化求解器，读取我们刚才配置的 ini 文件
    try:
        my_solver = Solver('baseline_propeller.ini')
    except Exception as e:
        print(f"读取配置文件失败，请检查 baseline.ini 是否在当前文件夹！错误信息: {e}")
        return

    print(f"\n物理环境初始化完毕：介质密度={my_solver.fluid.rho} kg/m^3, 转速={my_solver.rpm} RPM, 航速={my_solver.v_inf} m/s")
    print("开始执行 BEMT 气动力迭代求解...\n")

    # 核心求解运算
    T, Q, P, df_sections = my_solver.run()

    # 计算无量纲宏观性能系数
    J, CT, CQ, CP, eta = my_solver.rotor_coeffs(T, Q, P)

    print("="*45)
    print("【Table 1 基线螺旋桨核心性能评估报告】")
    print("-" * 45)
    print(f"实际测算总推力 (T)    : {T:.3f} N")
    print(f"原论文目标推力 (Target): 6.86 N")
    print(f"误差对比              : {abs(T - 6.86)/6.86 * 100:.2f} %")
    print("-" * 45)
    print(f"总消耗扭矩 (Q)        : {Q:.4f} N·m")
    print(f"总消耗功率 (P)        : {P:.2f} W")
    print("-" * 45)
    print(f"进速系数 (J)          : {J:.3f}")
    print(f"综合推进效率 (η)      : {eta * 100:.2f} %")
    print("="*45)

    # 导出各截面的详细参数用于论文图表
    df_sections.to_csv("Baseline_Section_Details.csv", index=False)
    print("\n* 提示: 沿径向各截面的详细气流偏转角、升阻力系数已保存至 Baseline_Section_Details.csv")

if __name__ == "__main__":
    evaluate_baseline()