import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # ← 加在最顶部

import numpy as np
import random
from main_optimization_dnn import evaluate_individual

class MOPSO:
    def __init__(self, num_particles, num_vars, bounds, max_iter):
        self.num_particles = num_particles
        self.num_vars = num_vars
        self.bounds = np.array(bounds)
        self.max_iter = max_iter
        
        # 1. 粒子群状态初始化 (随机生成)
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (num_particles, num_vars))
        
        # 【关键修改 A：注入火种】将第0号粒子强制设为你的全0基准翼型，确保至少有一个活着的向导
        self.positions[0] = np.zeros(num_vars)
        
        self.velocities = np.zeros((num_particles, num_vars))
        self.pbest_positions = self.positions.copy()
        self.pbest_objs = np.full((num_particles, 2), -np.inf) 
        self.current_objs = np.zeros((num_particles, 2))
        
        self.archive_positions = []
        self.archive_objs = []
        
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5

    def dominates(self, obj_A, obj_B):
        return np.all(obj_A >= obj_B) and np.any(obj_A > obj_B)

    def update_archive(self):
        # 【关键修改 B：淘汰死亡粒子】严格过滤，只有算出真实效率的粒子才有资格进入评比
        valid_mask = (self.current_objs[:, 0] != -np.inf) & (self.current_objs[:, 1] != -np.inf)
        valid_pos = self.positions[valid_mask]
        valid_objs = self.current_objs[valid_mask]
        
        # 如果当前代和历史存档全是废片，直接跳过
        if len(valid_objs) == 0 and len(self.archive_objs) == 0:
            return 
            
        # 与历史存档合并
        if len(self.archive_objs) > 0:
            all_objs = np.vstack([self.archive_objs, valid_objs]) if len(valid_objs) > 0 else np.array(self.archive_objs)
            all_pos = np.vstack([self.archive_positions, valid_pos]) if len(valid_pos) > 0 else np.array(self.archive_positions)
        else:
            all_objs = valid_objs
            all_pos = valid_pos
            
        # 帕累托非支配排序
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
            # 如果极端情况存档为空，优先选活着的历史最优
            valid_mask = self.pbest_objs[:, 0] != -np.inf
            if np.any(valid_mask):
                valid_pbests = self.pbest_positions[valid_mask]
                return valid_pbests[random.randint(0, len(valid_pbests) - 1)]
            return self.pbest_positions[random.randint(0, self.num_particles - 1)]
        
        idx = random.randint(0, len(self.archive_positions) - 1)
        return np.array(self.archive_positions[idx])

    def run(self):
        print("🚀 MOPSO 优化正式启动...")
        for iter_num in range(self.max_iter):
            alive_count = 0 # 统计每一代存活的合法翼型数量
            
            for i in range(self.num_particles):
                eta_air, eta_water = evaluate_individual(self.positions[i].tolist())
                self.current_objs[i] = [eta_air, eta_water]
                
                # 只有粒子存活，才更新个体最优
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
            print(f"========== 第 {iter_num + 1}/{self.max_iter} 代: 存活个体 {alive_count}/{self.num_particles}, 帕累托前沿已捕获 {len(self.archive_objs)} 个最优解 ==========")
            
            # 粒子群飞行更新
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
    bounds = [(-0.015, 0.015)] * 18
    mopso = MOPSO(num_particles=20, num_vars=18, bounds=bounds, max_iter=30)
    best_designs, best_efficiencies = mopso.run()

    print("\n帕累托前沿结果 (空气效率, 水下效率):")
    if len(best_efficiencies) == 0:
        print("未找到有效解，所有粒子均发散。")
    else:
        for eff in best_efficiencies:
            print(f"-> 空气: {eff[0]*100:.2f}% | 水下: {eff[1]*100:.2f}%")

        # 保存结果，防止结果丢失
        np.save('pareto_designs.npy', best_designs)
        np.save('pareto_efficiencies.npy', best_efficiencies)
        print("\n✅ 结果已保存到 pareto_designs.npy 和 pareto_efficiencies.npy")

        # 同时保存一份可读的txt
        with open('pareto_results.txt', 'w', encoding='utf-8') as f:
            f.write("帕累托前沿结果\n")
            f.write(f"Baseline: 空气 15.35% | 水下 31.01%\n")
            f.write("="*50 + "\n")
            for i, (eff, design) in enumerate(zip(best_efficiencies, best_designs)):
                f.write(f"解 {i+1}: 空气 {eff[0]*100:.2f}% | 水下 {eff[1]*100:.2f}%\n")
                f.write(f"  权重: {[round(w,6) for w in design]}\n")
        print("✅ 可读结果已保存到 pareto_results.txt")