import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import sys
import numpy as np

# 锁定工作目录（模块级别，最先执行）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_THIS_DIR)

# 将上级目录加入路径，以找到 pybemt 模块
sys.path.append(os.path.join(_THIS_DIR, '..'))

from pybemt.solver import Solver
from surrogate_model import AirfoilPredictor

# ==========================================
# 1. 全局单例：初始化 DNN 代理模型
# ==========================================
print("正在加载全局 DNN 代理模型与 Scaler...")
global_predictor = AirfoilPredictor(
    scaler_X_path='scaler_X.pkl',
    scaler_y_path='scaler_y.pkl',
    weights_path='airfoil_dnn_weights.pth'
)

# ==========================================
# 2. 安全的 DNN 包装器
# ==========================================
class DNNAirfoilWrapper:
    def __init__(self, predictor, weights, target_Re):
        self.predictor = predictor
        self.weights = weights
        self.target_Re = target_Re

    def Cl(self, alpha, Re=None, mach=0):
        alpha_deg = np.degrees(alpha)
        alpha_safe = float(np.clip(alpha_deg, 0.0, 10.0))
        cl, cd = self.predictor.predict(self.weights, self.target_Re, alpha_safe)
        return float(cl)

    def Cd(self, alpha, Re=None, mach=0):
        alpha_deg = np.degrees(alpha)
        alpha_safe = float(np.clip(alpha_deg, 0.0, 10.0))
        cl, cd = self.predictor.predict(self.weights, self.target_Re, alpha_safe)
        return max(float(cd), 0.0001)

# ==========================================
# 3. 个体评估函数
# ==========================================
def evaluate_individual(weights_list):
    try:
        # ---------- 空气工况 ----------
        solver_air = Solver('examples/baseline_propeller_air.ini')
        target_Re_air = 200000.0 if solver_air.fluid.rho > 500.0 else 75000
        for sec in solver_air.rotor.sections:
            sec.airfoil = DNNAirfoilWrapper(global_predictor, weights_list, target_Re_air)
        
        solver_air.run()  # 不接返回值
        
        # 从求解器对象读取结果
        T_air = solver_air.T
        P_air = solver_air.P
        V_air = solver_air.v_inf
        
        if P_air is None or P_air <= 0 or T_air is None:
            return -np.inf, -np.inf
        eta_air = float((T_air * V_air) / P_air)

        # ---------- 水下工况 ----------
        solver_water = Solver('examples/baseline_propeller_water.ini')
        target_Re_water = 200000.0 if solver_water.fluid.rho > 500.0 else 75000
        for sec in solver_water.rotor.sections:
            sec.airfoil = DNNAirfoilWrapper(global_predictor, weights_list, target_Re_water)
        
        solver_water.run()  # 不接返回值
        
        T_water = solver_water.T
        P_water = solver_water.P
        V_water = solver_water.v_inf
        
        if P_water is None or P_water <= 0 or T_water is None:
            return -np.inf, -np.inf
        eta_water = float((T_water * V_water) / P_water)

        # 有效性检查
        if not np.isfinite(eta_air) or eta_air <= 0:
            return -np.inf, -np.inf
        if not np.isfinite(eta_water) or eta_water <= 0:
            return -np.inf, -np.inf

        return eta_air, eta_water

    except Exception as e:
        import traceback
        traceback.print_exc()
        return -np.inf, -np.inf


# ==========================================
# 4. 本地测试
# ==========================================
if __name__ == '__main__':
    test_weights = [0.0] * 18
    print("\n--- 单样本测试启动 ---")
    eta_air, eta_water = evaluate_individual(test_weights)
    if eta_air != -np.inf:
        print(f"✅ 测试成功 -> 空气效率: {eta_air*100:.2f}%, 水下效率: {eta_water*100:.2f}%")
    else:
        print("❌ 测试失败，请检查上方报错信息")