import numpy as np
import random
from main_optimization_dnn import evaluate_individual
# 多目标粒子群优化（MOPSO）的主循环，负责在 21 维空间中生成、变异和筛选候选螺旋桨。
class MOPSO:
    def __init__(self, num_particles, num_vars, bounds, max_iter):
        self.num_particles = num_particles
        self.num_vars = num_vars
        self.bounds = np.array(bounds)
        self.max_iter = max_iter
        
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (num_particles, num_vars))
        
        # 强制将第0号粒子设为原文指定的基础桨参数 (cm=0.21, β=50, m=0.5, 翼型形变=0)
        self.positions[0] = np.array([0.21, 50.0, 0.5] + [0.0] * 18)
        
        self.velocities = np.zeros((num_particles, num_vars))
        self.pbest_positions = self.positions.copy()
        self.pbest_objs = np.full((num_particles, 2), -np.inf) 
        self.current_objs = np.zeros((num_particles, 2))
        self.archive_positions = []
        self.archive_objs = []
        
        self.w, self.c1, self.c2 = 0.5, 1.5, 1.5

    def dominates(self, obj_A, obj_B):
        return np.all(obj_A >= obj_B) and np.any(obj_A > obj_B)

    def update_archive(self):
        valid_mask = (self.current_objs[:, 0] != -np.inf) & (self.current_objs[:, 1] != -np.inf)
        valid_pos = self.positions[valid_mask]
        valid_objs = self.current_objs[valid_mask]
        
        if len(valid_objs) == 0 and len(self.archive_objs) == 0:
            return 
            
        if len(self.archive_objs) > 0:
            all_objs = np.vstack([self.archive_objs, valid_objs]) if len(valid_objs) > 0 else np.array(self.archive_objs)
            all_pos = np.vstack([self.archive_positions, valid_pos]) if len(valid_pos) > 0 else np.array(self.archive_positions)
        else:
            all_objs = valid_objs
            all_pos = valid_pos
            
        is_dominated = np.zeros(len(all_objs), dtype=bool)
        for i in range(len(all_objs)):
            for j in range(len(all_objs)):
                if i != j and self.dominates(all_objs[j], all_objs[i]):
                    is_dominated[i] = True
                    break
                    
        self.archive_positions = all_pos[~is_dominated].tolist()
        self.archive_objs = all_objs[~is_dominated].tolist()

    def get_gbest(self):
        if len(self.archive_positions) == 0:
            valid_mask = self.pbest_objs[:, 0] != -np.inf
            if np.any(valid_mask):
                valid_pbests = self.pbest_positions[valid_mask]
                return valid_pbests[random.randint(0, len(valid_pbests) - 1)]
            return self.pbest_positions[random.randint(0, self.num_particles - 1)]
        idx = random.randint(0, len(self.archive_positions) - 1)
        return np.array(self.archive_positions[idx])

    def run(self):
        print("🚀 21维跨介质 MOPSO 联合优化正式启动...")
        for iter_num in range(self.max_iter):
            alive_count = 0 
            
            for i in range(self.num_particles):
                eta_air, eta_water = evaluate_individual(self.positions[i].tolist())
                self.current_objs[i] = [eta_air, eta_water]
                
                if eta_air != -np.inf:
                    alive_count += 1
                    if self.pbest_objs[i][0] == -np.inf or self.dominates(self.current_objs[i], self.pbest_objs[i]):
                        self.pbest_objs[i] = self.current_objs[i].copy()
                        self.pbest_positions[i] = self.positions[i].copy()
                    elif not self.dominates(self.pbest_objs[i], self.current_objs[i]):
                        if random.random() < 0.5:
                            self.pbest_objs[i] = self.current_objs[i].copy()
                            self.pbest_positions[i] = self.positions[i].copy()
            
            self.update_archive()
            print(f"========== 第 {iter_num + 1}/{self.max_iter} 代: 存活个体 {alive_count}/{self.num_particles}, 帕累托库 {len(self.archive_objs)} 个解 ==========")
            
            for i in range(self.num_particles):
                gbest_pos = self.get_gbest()
                r1, r2 = random.random(), random.random()
                
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                                      self.c2 * r2 * (gbest_pos - self.positions[i]))
                
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])
                
        print("\n🎉 MOPSO 优化完成！")
        return np.array(self.archive_positions), np.array(self.archive_objs)

if __name__ == "__main__":
    # 前3维: cm (0.15~0.24), beta (2~100), m (0.3~0.5)
    bounds_3 = [(0.15, 0.24), (2.0, 100.0), (0.3, 0.5)]
    # 后18维: 翼型控制点
    bounds_18 = [(-0.015, 0.015)] * 18  
    # 组合成完整的 21 维边界
    full_bounds = bounds_3 + bounds_18
    
    mopso = MOPSO(num_particles=20, num_vars=21, bounds=full_bounds, max_iter=30)
    best_designs, best_efficiencies = mopso.run()
    
    print("\n帕累托前沿结果 (空气效率, 水下效率):")
    if len(best_efficiencies) == 0:
        print("未找到有效解，所有粒子均发散。")
    else:
        for eff in best_efficiencies:
            print(f"-> 空气: {eff[0]*100:.2f}% | 水下: {eff[1]*100:.2f}%")