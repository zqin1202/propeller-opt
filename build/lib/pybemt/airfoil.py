import numpy as np
# 导入你的 AI 预测器
from .surrogate_model import AirfoilPredictor

# 【极其关键】在模块最外层实例化预测器，确保全局只加载一次！
# 这样在成千上万次迭代中，才不会重复读取 20 万行的 CSV 文件。
AI_PREDICTOR = AirfoilPredictor(
    csv_path='pybemt/Airfoil_Aerodynamic_Database.csv', 
    weights_path='pybemt/airfoil_dnn_weights.pth'
)

class Airfoil:
    """
    被 AI 代理模型接管的全新 Airfoil 类
    """
    def __init__(self, name, fluid=None):
        self.name = name
        self.fluid = fluid
        # 默认的 18 个形状权重（未变形时的 CLARKY 基准面，全为0）
        # 后续遗传算法优化时，会直接修改这个属性！
        self.weights = [0.0] * 18 

    def get_cl_cd(self, alpha, Re=None):
        """
        BEMT 求解器会疯狂调用这个函数。
        :param alpha: 求解器传来的迎角 (单位：弧度 !!!)
        :param Re: 求解器传来的雷诺数
        """
        # 1. 弧度转角度 (Radians to Degrees)
        alpha_deg = np.degrees(alpha)
        
        # 2. 物理边界安全限幅器 (防止 BEMT 迭代初期迎角飞出 DNN 认知范围)
        # 你的模型训练范围是 0~10度，这里我们稍微放宽一点点，限制在 0~10度 之间
        alpha_deg_clipped = np.clip(alpha_deg, 0.0, 10.0)
        
        # 如果没有传入 Re，给一个默认值
        if Re is None:
            Re = 200000.0
            
        # 3. 呼叫 AI 进行毫秒级预测
        cl, cd = AI_PREDICTOR.predict(self.weights, Re, alpha_deg_clipped)
        
        return cl, cd