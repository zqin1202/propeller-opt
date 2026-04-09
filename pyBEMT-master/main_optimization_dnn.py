import warnings
warnings.filterwarnings("ignore")
import os
import sys
import numpy as np
import pandas as pd
import contextlib

# ==========================================
# Airfoil Geometry Constraint Utilities
# ==========================================

def count_stationary_points(y):
    """
    计算函数的驻点数量 (dy/dx = 0)
    """
    dy = np.gradient(y)
    sign_change = np.diff(np.sign(dy))
    return np.sum(sign_change != 0)


def curvature_penalty(y):
    """
    曲率计算，用于检测波浪形翼型
    """
    d2y = np.gradient(np.gradient(y))
    return np.max(np.abs(d2y))


def geometry_constraints(x, yu, yl, yu_ref=None, yl_ref=None):
    """
    几何合法性检查

    返回:
    True -> 几何合法
    False -> 非法翼型
    """

    # --------------------------------
    # 1 上下表面不能相交
    # --------------------------------
    if np.any(yu <= yl):
        return False

    # --------------------------------
    # 2 厚度约束
    # --------------------------------
    t = yu - yl

    if np.min(t) < 0.01:
        return False

    # --------------------------------
    # 3 尾缘厚度
    # --------------------------------
    if t[-1] < 0.002:
        return False

    # --------------------------------
    # 4 平滑性约束 (stationary points)
    # --------------------------------
    if count_stationary_points(yu) != 1:
        return False

    if count_stationary_points(yl) > 2:
        return False

    # --------------------------------
    # 5 曲率约束 (防止波浪)
    # --------------------------------
    if curvature_penalty(yu) > 5:
        return False

    if curvature_penalty(yl) > 5:
        return False

    return True

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pybemt.solver import Solver
from surrogate_model import AirfoilPredictor

print("正在加载全局 DNN 代理模型与 Scaler...")
global_predictor = AirfoilPredictor(
    scaler_X_path='scaler_X.pkl', 
    scaler_y_path='scaler_y.pkl', 
    weights_path='airfoil_dnn_weights.pth'
)


class DNNAirfoilWrapper:
    def __init__(self, predictor, weights, target_Re):
        self.predictor = predictor
        self.weights = weights
        self.target_Re = target_Re
        
    def Cl(self, alpha, Re=None, mach=0):
        alpha_deg = np.degrees(alpha)
        # 物理惩罚：如果迎角严重小于0，给它一个很小的升力甚至负升力
        if alpha_deg < 0.0:
            return -0.2  # 人为返回负升力，让算法知道这不合理
        if alpha_deg > 15.0:
            return 0.1   # 大迎角失速惩罚
            
        alpha_safe = float(np.clip(alpha_deg, 0.0, 10.0))
        cl, cd = self.predictor.predict(self.weights, self.target_Re, alpha_safe)
        return float(cl)
        
    def Cd(self, alpha, Re=None, mach=0):
        alpha_deg = np.degrees(alpha)
        # 物理惩罚：如果是负迎角或极大迎角，阻力飙升
        if alpha_deg < 0.0 or alpha_deg > 15.0:
            return 0.5  # 巨大的阻力惩罚
            
        alpha_safe = float(np.clip(alpha_deg, 0.0, 10.0))
        cl, cd = self.predictor.predict(self.weights, self.target_Re, alpha_safe)
        return max(float(cd), 0.0001)

def calculate_actual_chord(radius, R, cm, beta, m):
    """
    根据给定的三维参数，计算特定截面的真实物理弦长
    输入:
        radius (float): 当前叶素截面的局部半径 r
        R (float): 螺旋桨总半径
        cm (float): 最大无量纲弦长 (0.15~0.24)
        beta (float): 弦长分布系数 (2~100)
        m (float): 最大弦长所在位置 (0.3~0.5)
    输出:
        actual_chord (float): 当前截面的物理真实弦长 c
    """
    xi = radius / R  # 计算无量纲径向坐标 ξ
    
    c_nondim = cm * (beta ** (-(xi - m)**2)) 
    
    # 还原为物理弦长
    actual_chord = c_nondim * R
    return float(actual_chord)

def evaluate_individual(weights_list):
    try:
        cm, beta, m = weights_list[0], weights_list[1], weights_list[2]
        weights_18 = weights_list[3:21]  # 严格提取中间18个
        pitch_scale = weights_list[21]   # 提取最后一个作为缩放系数

        # ---------------- 考核 A: 空气环境 ----------------
        solver_air = Solver('examples/baseline_propeller_air.ini')
        target_Re_air = 200000.0 if solver_air.fluid.rho > 500.0 else 50000.0
        R_propeller = solver_air.rotor.diameter / 2.0

        for sec in solver_air.rotor.sections:
            sec.airfoil = DNNAirfoilWrapper(global_predictor, weights_18, target_Re_air)
            sec.chord = calculate_actual_chord(sec.radius, R_propeller, cm, beta, m)
            # 动态缩放扭转角，并强制同步到底层弧度
            sec.pitch = sec.pitch * pitch_scale  
            if hasattr(sec, 'pitch_rad'):
                sec.pitch_rad = np.radians(sec.pitch)

        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            T_air, Q_air, P_air, df_air = solver_air.run()
            
        # 【找回丢失的代码】计算空气效率
        _, _, _, _, eta_air = solver_air.rotor_coeffs(T_air, Q_air, P_air)

        # ---------------- 考核 B: 水下环境 ----------------
        solver_water = Solver('examples/baseline_propeller_water.ini')
        target_Re_water = 200000.0 if solver_water.fluid.rho > 500.0 else 50000.0

        for sec in solver_water.rotor.sections:
            sec.airfoil = DNNAirfoilWrapper(global_predictor, weights_18, target_Re_water)
            sec.chord = calculate_actual_chord(sec.radius, R_propeller, cm, beta, m)
            # 水下同样进行缩放和同步
            sec.pitch = sec.pitch * pitch_scale  
            if hasattr(sec, 'pitch_rad'):
                sec.pitch_rad = np.radians(sec.pitch)

        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            T_water, Q_water, P_water, df_water = solver_water.run()
            
        # 【找回丢失的代码】计算水下效率
        _, _, _, _, eta_water = solver_water.rotor_coeffs(T_water, Q_water, P_water)

        # 剔除无效效率
        if np.isnan(eta_air) or np.isnan(eta_water) or eta_air <= 0 or eta_water <= 0:
            return 0.0, 0.0

        return float(eta_air), float(eta_water)

    except Exception as e:
        # 强制报错显形
        print(f" [致命错误] evaluate_individual 发生代码崩溃: {e}") 
        return 0.0, 0.0


def print_optimal_details(weights_list):
    """
    专门用于在优化结束后，解剖并打印最佳螺旋桨的 3D 截面细节（攻角、升阻力分布）
    """
    cm, beta, m = weights_list[0], weights_list[1], weights_list[2]
    weights_18 = weights_list[3:21]
    pitch_scale = weights_list[21]

    print("\n" + "="*60)
    print("🔬 [深度解剖] 最佳两栖螺旋桨 沿展向截面气动/水动力分布")
    print("="*60)

    # --- 空气环境解剖 ---
    solver_air = Solver('examples/baseline_propeller_air.ini')
    target_Re_air = 200000.0 if solver_air.fluid.rho > 500.0 else 50000.0
    R_propeller = solver_air.rotor.diameter / 2.0
    
    for sec in solver_air.rotor.sections:
        sec.airfoil = DNNAirfoilWrapper(global_predictor, weights_18, target_Re_air)
        sec.chord = calculate_actual_chord(sec.radius, R_propeller, cm, beta, m)
        # 【修复遗漏】解剖打印时也要应用缩放！
        sec.pitch = sec.pitch * pitch_scale
        if hasattr(sec, 'pitch_rad'):
            sec.pitch_rad = np.radians(sec.pitch)
        
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
        _, _, _, df_air = solver_air.run()
        
    print("\n【工况 A: 空气中 (7200 RPM, 2.0 m/s)】")
    if 'alpha' in df_air.columns:
        df_air['alpha_deg'] = np.degrees(df_air['alpha'])
        print(df_air[['radius', 'chord', 'pitch', 'alpha_deg', 'Cl', 'Cd']].to_string(index=False, float_format="%.4f"))
    else:
        print(df_air.to_string(index=False))

    # --- 水下环境解剖 ---
    solver_water = Solver('examples/baseline_propeller_water.ini')
    target_Re_water = 200000.0 if solver_water.fluid.rho > 500.0 else 50000.0
    
    for sec in solver_water.rotor.sections:
        sec.airfoil = DNNAirfoilWrapper(global_predictor, weights_18, target_Re_water)
        sec.chord = calculate_actual_chord(sec.radius, R_propeller, cm, beta, m)
        # 【修复遗漏】解剖打印时也要应用缩放！
        sec.pitch = sec.pitch * pitch_scale
        if hasattr(sec, 'pitch_rad'):
            sec.pitch_rad = np.radians(sec.pitch)
        
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
        _, _, _, df_water = solver_water.run()
        
    print("\n【工况 B: 水下 (900 RPM, 2.0 m/s)】")
    if 'alpha' in df_water.columns:
        df_water['alpha_deg'] = np.degrees(df_water['alpha'])
        print(df_water[['radius', 'chord', 'pitch', 'alpha_deg', 'Cl', 'Cd']].to_string(index=False, float_format="%.4f"))
    else:
        print(df_water.to_string(index=False))

        
if __name__ == '__main__':
    # 基线螺旋桨参数
    baseline_params = [0.21, 50.0, 0.5] + [0.0] * 18
    
    print("\n" + "="*50)
    print("🚀 开始对 [基线螺旋桨] 进行独立双介质评估...")
    print("="*50)
    
    # 临时修改：不使用隐藏打印的包裹，让 pyBEMT 自带的推力/扭矩打印出来
    # （你需要去 evaluate_individual 函数里，把 with HiddenPrints(): 这一行临时注释掉，
    # 或者直接看这里的宏观结果）
    
    # ---------------- 考核空气 ----------------
    solver_air = Solver('examples/baseline_propeller_air.ini')
    target_Re_air = 200000.0 if solver_air.fluid.rho > 500.0 else 50000.0
    R_propeller = solver_air.rotor.diameter / 2.0
    for sec in solver_air.rotor.sections:
        sec.airfoil = DNNAirfoilWrapper(global_predictor, baseline_params[3:], target_Re_air)
        sec.chord = calculate_actual_chord(sec.radius, R_propeller, *baseline_params[:3])
        
    print("\n[空气工况评估结果]")
    T_air, Q_air, P_air, _ = solver_air.run()
    J_air, CT_air, CQ_air, CP_air, eta_air = solver_air.rotor_coeffs(T_air, Q_air, P_air)
    print(f"推力(T): {T_air:.4f} N, 扭矩(Q): {Q_air:.4f} Nm, 功率(P): {P_air:.4f} W")
    print(f"进位比(J): {J_air:.4f}, 效率: {eta_air*100:.2f}%")

    # ---------------- 考核水下 ----------------
    solver_water = Solver('examples/baseline_propeller_water.ini')
    target_Re_water = 200000.0 if solver_water.fluid.rho > 500.0 else 50000.0
    for sec in solver_water.rotor.sections:
        sec.airfoil = DNNAirfoilWrapper(global_predictor, baseline_params[3:], target_Re_water)
        sec.chord = calculate_actual_chord(sec.radius, R_propeller, *baseline_params[:3])

    print("\n[水下工况评估结果]")
    T_water, Q_water, P_water, df_water = solver_water.run()
    J_water, CT_water, CQ_water, CP_water, eta_water = solver_water.rotor_coeffs(T_water, Q_water, P_water)
    print(f"推力(T): {T_water:.4f} N, 扭矩(Q): {Q_water:.4f} Nm, 功率(P): {P_water:.4f} W")
    print(f"进位比(J): {J_water:.4f}, 效率: {eta_water*100:.2f}%")