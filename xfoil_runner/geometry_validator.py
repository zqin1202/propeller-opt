import numpy as np

class GeometryValidator:
    def __init__(self, base_dat_path):
        """
        初始化时读取基准翼型，并生成标准化的网格和基准厚度。
        """
        self.x_std = None
        self.y_up_base = None
        self.y_low_base = None
        self.t_base = None
        self._load_and_standardize(base_dat_path)

    def _load_and_standardize(self, filepath):
        """针对你提供的特定格式 (LE->TE Upper, LE->TE Lower) 解析文件"""
        x_up, y_up = [], []
        x_low, y_low = [], []
        
        is_upper = True
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'CLARKY' in line.upper():
                    continue
                parts = line.split()
                if len(parts) == 2:
                    x_val, y_val = float(parts[0]), float(parts[1])
                    
                    # 识别从上表面切换到下表面的分割点：x 突然变回 0.0
                    if i > 1 and x_val == 0.0 and y_val == 0.0:
                        is_upper = False
                        
                    if is_upper:
                        x_up.append(x_val)
                        y_up.append(y_val)
                    else:
                        x_low.append(x_val)
                        y_low.append(y_val)
                        
        # 强制将 x_up 和 x_low 转换为 numpy 数组
        x_up, y_up = np.array(x_up), np.array(y_up)
        x_low, y_low = np.array(x_low), np.array(y_low)

        # 建立标准化的 x 网格 (100个节点，余弦分布以加密前后缘)
        beta = np.linspace(0, np.pi, 100)
        self.x_std = 0.5 * (1 - np.cos(beta))
        
        # 插值得到基准上下表面
        self.y_up_base = np.interp(self.x_std, x_up, y_up)
        self.y_low_base = np.interp(self.x_std, x_low, y_low)
        self.t_base = self.y_up_base - self.y_low_base

    def _apply_hicks_henne(self, alphas_up, alphas_low):
        """应用 Hicks-Henne 变形"""
        x_c = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9])
        y_up_def = np.copy(self.y_up_base)
        y_low_def = np.copy(self.y_low_base)
        
        # 避免 log(0)
        x_safe = np.clip(self.x_std, 1e-6, 1.0)
        
        for i in range(9):
            bump = np.sin(np.pi * x_safe ** (np.log(0.5) / np.log(x_c[i]))) ** 2
            y_up_def += alphas_up[i] * bump
            y_low_def += alphas_low[i] * bump
            
        return y_up_def, y_low_def

    def _count_stationary_points(self, y_coords):
        """计算驻点数量，判断平滑度"""
        dy = np.diff(y_coords)
        dy_non_zero = dy[dy != 0]
        sign_changes = np.sum(np.diff(np.sign(dy_non_zero)) != 0)
        return sign_changes

    def check_geometry(self, alphas):
        """
        对外接口：传入18个变形参数，返回 True (合法) 或 False (畸形)
        alphas: list or array of length 18
        """
        if len(alphas) != 18:
            raise ValueError("需要18个Hicks-Henne参数")
            
        y_up_def, y_low_def = self._apply_hicks_henne(alphas[:9], alphas[9:])
        t_def = y_up_def - y_low_def
        
        # 1. 厚度约束 (Table 6) + 绝对最小厚度防御
        min_absolute_thickness = 0.001  # 防止尾缘交叉的最后防线
        lower_bound = np.maximum(0.6 * self.t_base, min_absolute_thickness)
        upper_bound = 1.2 * self.t_base
        
        if np.any(t_def < lower_bound) or np.any(t_def > upper_bound):
            return False
            
        # 2. 平滑度约束 (Table 6)
        stat_u = self._count_stationary_points(y_up_def)
        stat_l = self._count_stationary_points(y_low_def)
        
        if stat_u != 1:
            return False
        if stat_l > 2:
            return False
            
        return True