import numpy as np

class GeometryValidator:
    def __init__(self, base_dat_path):
        """初始化时预计算 Clark-Y 的基础包络和 Hicks-Henne 凸块矩阵"""
        x_up, y_up = [], []
        x_low, y_low = [], []
        is_upper = True
        
        with open(base_dat_path, 'r') as f:
            for i, line in enumerate(f):
                if 'CLARKY' in line.upper(): continue
                parts = line.split()
                if len(parts) == 2:
                    try:
                        x_val, y_val = float(parts[0]), float(parts[1])
                        if i > 2 and x_val == 0.0 and y_val == 0.0:
                            is_upper = False
                        
                        if is_upper:
                            x_up.append(x_val); y_up.append(y_val)
                        else:
                            x_low.append(x_val); y_low.append(y_val)
                    except ValueError:
                        continue
                        
        x_up_raw, y_up_raw = np.array(x_up), np.array(y_up)
        x_low_raw, y_low_raw = np.array(x_low), np.array(y_low)
        
        beta = np.linspace(0, np.pi, 100)
        self.x_std = 0.5 * (1 - np.cos(beta))
        
        self.y_up_base = np.interp(self.x_std, x_up_raw, y_up_raw)
        self.y_low_base = np.interp(self.x_std, x_low_raw, y_low_raw)
        self.t_base = self.y_up_base - self.y_low_base
        
        peak_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.bumps = np.zeros((9, 100))
        x_safe = np.clip(self.x_std, 1e-6, 1.0)
        for i in range(9):
            m_i = np.log(0.5) / np.log(peak_x[i])
            self.bumps[i, :] = (np.sin(np.pi * x_safe**m_i))**3

    def is_valid(self, weights_18, debug_print=False):
        """对传入的 18 维几何参数进行生死判决"""
        w_up = weights_18[:9]
        w_low = weights_18[9:]
        
        y_up = self.y_up_base + np.dot(w_up, self.bumps)
        y_low = self.y_low_base + np.dot(w_low, self.bumps)
        t_new = y_up - y_low
        
        # --- 1. 严格厚度约束 (Table 6) ---
        # 切片 [1:] 的作用是跳过前缘点 (x=0, 厚度为0)，防止因为数值计算导致除零或误判
        lower_bound = 0.6 * self.t_base
        upper_bound = 1.2 * self.t_base
        
        if not np.all((t_new[1:] >= lower_bound[1:]) & (t_new[1:] <= upper_bound[1:])):
            if debug_print: print(" [拦截] 翼型厚度未处于 0.6 ~ 1.2 倍基准厚度之间")
            return False
            
        # 物理防线：绝对不允许任何区域（除前缘外）厚度小于或等于 0
        if np.any(t_new[1:] <= 0.0):
            if debug_print: print(" [拦截] 发生致命的物理交叉 (上下表面重合或反转)")
            return False
            
        # --- 2. 光顺度约束 (Table 6) ---
        # 提升容差到 5e-4，防止插值曲线的微观锯齿干扰驻点判定
        tolerance = 5e-4 
        
        # 上表面驻点数量 == 1
        dy_up = np.diff(y_up)
        dy_up[np.abs(dy_up) < tolerance] = 0 
        sign_dy_up = np.sign(dy_up)
        sign_dy_up = sign_dy_up[sign_dy_up != 0] 
        stat_up = np.sum(sign_dy_up[:-1] != sign_dy_up[1:])
        if stat_up != 1:
            if debug_print: print(f" [拦截] 上表面驻点数量异常: {stat_up} (要求=1)")
            return False
            
        # 下表面驻点数量 <= 2
        dy_low = np.diff(y_low)
        dy_low[np.abs(dy_low) < tolerance] = 0
        sign_dy_low = np.sign(dy_low)
        sign_dy_low = sign_dy_low[sign_dy_low != 0]
        stat_low = np.sum(sign_dy_low[:-1] != sign_dy_low[1:])
        if stat_low > 2:
            if debug_print: print(f" [拦截] 下表面驻点数量异常: {stat_low} (要求<=2)")
            return False
            
        return True

# ==========================================
# 独立测试模块：尺子精度自检
# ==========================================
if __name__ == "__main__":
    import os
    import sys
    # 填入相对路径，用于自检
    test_path = os.path.join(os.path.dirname(__file__), "..", "xfoil_runner", "CLARKY_geo.dat")
    if not os.path.exists(test_path):
        print(f"找不到测试文件: {test_path}")
        sys.exit()
        
    validator = GeometryValidator(test_path)
    baseline_params = np.zeros(18)
    print("正在测试基准 CLARKY 翼型是否能通过验证器...")
    result = validator.is_valid(baseline_params, debug_print=True)
    print(f"自检结果: {'通过 (True)' if result else '失败 (False)'}")