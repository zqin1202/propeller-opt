import numpy as np
import random
from main_optimization_dnn import evaluate_individual

class PSO:
    def __init__(self, num_particles, num_vars, bounds, max_iter, weight_air=0.5, weight_water=0.5):
        self.num_particles = num_particles
        self.num_vars = num_vars
        self.bounds = np.array(bounds)
        self.max_iter = max_iter
        
        # 优化权重设置
        self.w_air = weight_air
        self.w_water = weight_water
        
        # 初始化粒子群位置
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (num_particles, num_vars))
        # 【火种注入】：将第0号粒子设为基线螺旋桨，保证种群初期有正确的进化方向
        self.positions[0] = np.array([0.21, 50.0, 0.5] + [0.0] * 18)
        
        self.velocities = np.zeros((num_particles, num_vars))
        
        # 个体历史最优
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(num_particles, -np.inf)
        
        # 全局历史最优
        self.gbest_position = self.positions[0].copy()
        self.gbest_fitness = -np.inf
        
        self.w, self.c1, self.c2 = 0.5, 1.5, 1.5

    def run(self):
        print(f"🚀 21维跨介质 PSO 单目标联合优化正式启动...")
        print(f"🎯 优化目标函数: Fitness = {self.w_air} * 空气效率 + {self.w_water} * 水中效率\n")
        
        for iter_num in range(self.max_iter):
            alive_count = 0 
            
            for i in range(self.num_particles):
                # 评估水/空效率
                eta_air, eta_water = evaluate_individual(self.positions[i].tolist())
                
                # 只有当算出了真实效率（大于0），才计算适应度
                if eta_air > 0 and eta_water > 0:
                    alive_count += 1
                    fitness = self.w_air * eta_air + self.w_water * eta_water
                    
                    # 更新个体最优
                    if fitness > self.pbest_fitness[i]:
                        self.pbest_fitness[i] = fitness
                        self.pbest_positions[i] = self.positions[i].copy()
                        
                    # 更新全局最优
                    if fitness > self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.positions[i].copy()
            
            print(f"========== 第 {iter_num + 1}/{self.max_iter} 代: 存活个体 {alive_count}/{self.num_particles} ==========")
            if self.gbest_fitness != -np.inf:
                print(f"当前全局最高加权得分: {self.gbest_fitness:.4f}")
            
            # 粒子群飞行演化
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                                      self.c2 * r2 * (self.gbest_position - self.positions[i]))
                
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])
                
        print("\n🎉 PSO 优化完成！")
        return self.gbest_position, self.gbest_fitness

if __name__ == "__main__":
    # 物理参数边界设定
    bounds_3 = [(0.15, 0.24), (2.0, 100.0), (0.3, 0.5)]
    bounds_18 = [(-0.015, 0.015)] * 18  
    full_bounds = bounds_3 + bounds_18
    
    # 实例化 PSO (可通过 weight_air 和 weight_water 轻松调节优化侧重点)
    pso = PSO(num_particles=20, num_vars=21, bounds=full_bounds, max_iter=30, weight_air=0.5, weight_water=0.5)
    
    best_position, best_fitness = pso.run()
    
    print("\n" + "="*40)
    print("🏆 最终优化结果报告 🏆")
    print("="*40)
    
    if best_fitness == -np.inf:
        print("未找到有效解，所有粒子均未收敛。")
    else:
        print(f"最佳综合加权适应度: {best_fitness:.4f}")
        
        # 提取最佳水/空各自效率
        final_eta_air, final_eta_water = evaluate_individual(best_position.tolist())
        print(f"-> 对应空气中效率: {final_eta_air * 100:.2f}%")
        print(f"-> 对应水中效率:   {final_eta_water * 100:.2f}%")
        
        print("\n[最佳 21 维参数解]")
        print(f"弦长分布参数: cm = {best_position[0]:.4f}, beta = {best_position[1]:.4f}, m = {best_position[2]:.4f}")
        print("翼型形变权重 (18维):")
        print(np.round(best_position[3:], 4))