import numpy as np
import random
import os
import sys


# 导入评估函数
from main_optimization_dnn import evaluate_individual, print_optimal_details
# 导入独立出来的几何宪兵模块
from geometry_validator import GeometryValidator

# ==========================================
# 经典 PSO 主循环
# ==========================================
class PSO:
    def __init__(self, num_particles, num_vars, bounds, max_iter, validator, weight_air=0.5, weight_water=0.5):
        self.num_particles = num_particles
        self.num_vars = num_vars
        self.bounds = np.array(bounds)
        self.max_iter = max_iter
        self.validator = validator # 注入几何宪兵
        
        self.w_air = weight_air
        self.w_water = weight_water
        
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (num_particles, num_vars))
        self.velocities = np.zeros((num_particles, num_vars))
        
        # 强制设入火种，保证种群初期不全灭
        self.positions[0] = np.array([0.21, 50.0, 0.5] + [0.0] * 18)
        
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(num_particles, -np.inf)
        self.gbest_position = self.positions[0].copy()
        self.gbest_fitness = -np.inf
        
        self.w, self.c1, self.c2 = 0.5, 1.5, 1.5

    def run(self):
        print(f"🚀 21维跨介质 PSO (厚度/光顺度双约束版) 启动...")
        
        for iter_num in range(self.max_iter):
            alive_count = 0 
            
            for i in range(self.num_particles):
                # ====================================================
                # 【第一道防线】几何形态验证：非法翼型直接淘汰！
                # ====================================================
                if not self.validator.is_valid(self.positions[i][3:]):
                    continue 
                
                # ====================================================
                # 【第二道防线】物理求解评估：合法翼型进入 BEMT 考核
                # ====================================================
                eta_air, eta_water = evaluate_individual(self.positions[i].tolist())
                
                # 如果效率计算失败或无效 (<=0 或是 NaN)，淘汰
                if eta_air <= 0 or eta_water <= 0 or np.isnan(eta_air) or np.isnan(eta_water):
                    continue
                
                # ====================================================
                # 【存活与进化】两道防线均通过，计算适应度并更新历史最优
                # ====================================================
                alive_count += 1
                fitness = self.w_air * eta_air + self.w_water * eta_water
                
                # 更新个体历史最优 (pbest)
                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                    
                # 更新全局历史最优 (gbest)
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
            
            # --- 每一代结束后的战报输出 ---
            print(f"========== 第 {iter_num + 1}/{self.max_iter} 代: 存活且合规个体 {alive_count}/{self.num_particles} ==========")
            if self.gbest_fitness != -np.inf:
                print(f"当前全局最高加权得分: {self.gbest_fitness:.4f} (对应气动效率: {eta_air:.4f}, 水动效率: {eta_water:.4f})")
            
            # --- 粒子群位置与速度更新 (飞行演化) ---
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                                      self.c2 * r2 * (self.gbest_position - self.positions[i]))
                
                self.positions[i] = self.positions[i] + self.velocities[i]
                # 单体硬边界限幅 (严守搜索空间)
                self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])
                
        print("\n🎉 PSO 优化完成！")
        return self.gbest_position, self.gbest_fitness 


if __name__ == "__main__":
    # ⚠️ 【极其重要】请把这里的相对路径修改为你电脑里 CLARKY_geo.dat 的真实位置！
    base_airfoil_path = os.path.join(os.path.dirname(__file__), "..", "xfoil_runner", "CLARKY_geo.dat")
    
    if not os.path.exists(base_airfoil_path):
        print(f"❌ 找不到基础翼型文件: {base_airfoil_path}")
        print("请检查你的路径配置！")
        sys.exit()

    # 初始化几何宪兵
    geom_validator = GeometryValidator(base_airfoil_path)
    
    # 物理参数边界设定
    bounds_3 = [(0.15, 0.24), (2.0, 100.0), (0.3, 0.5)]
    bounds_18 = [(-0.01, 0.01)] * 18  # 保持 ±0.01
    full_bounds = bounds_3 + bounds_18
    
    # 启动 PSO
    pso = PSO(num_particles=20, num_vars=21, bounds=full_bounds, max_iter=30, validator=geom_validator)
    best_position, best_fitness = pso.run()
    
    print("\n" + "="*40)
    print("🏆 最终优化结果报告 🏆")
    print("="*40)
    
    if best_fitness == -np.inf:
        print("未找到有效解，所有粒子均未收敛。")
    else:
        print(f"最佳综合加权适应度: {best_fitness:.4f}")
        final_eta_air, final_eta_water = evaluate_individual(best_position.tolist())
        print(f"-> 对应空气中效率: {final_eta_air * 100:.2f}%")
        print(f"-> 对应水中效率:   {final_eta_water * 100:.2f}%\n")
        
        print("[最佳 21 维参数解]")
        print(f"三维弦长参数: cm = {best_position[0]:.4f}, beta = {best_position[1]:.4f}, m = {best_position[2]:.4f}")
        print("二维翼型权重 (18维):")
        print(np.round(best_position[3:], 4))
        
        # 打印详细解剖数据
        print_optimal_details(best_position.tolist())