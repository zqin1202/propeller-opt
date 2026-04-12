import numpy as np
import random
import os
import sys
import torch
from multiprocessing import Pool, cpu_count


# 导入评估函数
from main_optimization_dnn import evaluate_individual, print_optimal_details
# 导入独立出来的几何宪兵模块
from geometry_validator import GeometryValidator


# ==========================================
# 多进程评估包装函数
# ==========================================
# 全局变量用于进程初始化
_worker_predictor = None
_worker_solver_air = None
_worker_solver_water = None
_worker_initialized = False


def init_worker():
    """
    工作进程初始化函数
    确保每个子进程只加载一次模型，避免重复加载和重复打印
    """
    global _worker_predictor, _worker_solver_air, _worker_solver_water, _worker_initialized

    if _worker_initialized:
        return  # 已经初始化过，直接返回

    try:
        # 抑制打印信息
        import sys
        import io
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # 加载模型
        from surrogate_model import AirfoilPredictor
        from pybemt.solver import Solver

        _worker_predictor = AirfoilPredictor(
            scaler_X_path='scaler_X.pkl',
            scaler_y_path='scaler_y.pkl',
            weights_path='airfoil_dnn_weights.pth'
        )

        # 加载Solver对象
        _worker_solver_air = Solver('examples/baseline_propeller_air.ini')
        _worker_solver_water = Solver('examples/baseline_propeller_water.ini')

        # 恢复标准输出
        sys.stdout = original_stdout

        _worker_initialized = True
    except Exception as e:
        # 恢复标准输出并打印错误
        sys.stdout = original_stdout
        print(f"工作进程初始化失败: {e}")


def evaluate_particle_optimized(params_tuple):
    """
    优化版本的粒子评估函数，使用21维参数
    Args:
        params_tuple: 粒子参数的元组形式
    Returns:
        (eta_air, eta_water): 气动和水动效率
    """
    global _worker_predictor, _worker_solver_air, _worker_solver_water

    try:
        position_list = list(params_tuple)
        cm, beta, m = position_list[0], position_list[1], position_list[2]
        weights_18 = position_list[3:21]

        import numpy as np
        import contextlib
        import os

        # 约束：18维翼型权重 L1 范数 ≤ 0.12
        if np.sum(np.abs(weights_18)) > 0.12:
            return 0.0, 0.0

        # 空气环境评估
        target_Re_air = 200000.0 if _worker_solver_air.fluid.rho > 500.0 else 50000.0
        R_propeller = _worker_solver_air.rotor.diameter / 2.0

        # 应用翼型与弦长，保留配置文件中的原始 pitch
        for sec in _worker_solver_air.rotor.sections:
            from main_optimization_dnn import DNNAirfoilWrapper
            sec.airfoil = DNNAirfoilWrapper(_worker_predictor, weights_18, target_Re_air)
            sec.chord = calculate_actual_chord(sec.radius, R_propeller, cm, beta, m)

        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            T_air, Q_air, P_air, df_air = _worker_solver_air.run()

        _, _, _, _, eta_air = _worker_solver_air.rotor_coeffs(T_air, Q_air, P_air)

        # 水下环境评估
        target_Re_water = 200000.0 if _worker_solver_water.fluid.rho > 500.0 else 50000.0

        # 应用翼型与弦长，保留配置文件中的原始 pitch
        for sec in _worker_solver_water.rotor.sections:
            from main_optimization_dnn import DNNAirfoilWrapper
            sec.airfoil = DNNAirfoilWrapper(_worker_predictor, weights_18, target_Re_water)
            sec.chord = calculate_actual_chord(sec.radius, R_propeller, cm, beta, m)

        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
            T_water, Q_water, P_water, df_water = _worker_solver_water.run()

        _, _, _, _, eta_water = _worker_solver_water.rotor_coeffs(T_water, Q_water, P_water)

        if np.isnan(eta_air) or np.isnan(eta_water) or eta_air <= 0 or eta_water <= 0:
            return 0.0, 0.0

        return float(eta_air), float(eta_water)

    except Exception as e:
        return 0.0, 0.0


def calculate_actual_chord(radius, R, cm, beta, m):
    """计算实际弦长"""
    xi = radius / R
    c_nondim = cm * (beta ** (-(xi - m)**2))
    actual_chord = c_nondim * R
    return float(actual_chord)


def evaluate_particle(params_tuple):
    """
    包装函数，支持多进程调用evaluate_individual
    Args:
        params_tuple: 粒子参数的元组形式
    Returns:
        (eta_air, eta_water): 气动和水动效率
    """
    from main_optimization_dnn import evaluate_individual
    position_list = list(params_tuple)
    return evaluate_individual(position_list)

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

        # 强制设入火种，保证种群初期不全灭（21维版本）
        self.positions[0] = np.array([0.21, 50.0, 0.5] + [0.0] * 18)

        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(num_particles, -np.inf)
        self.gbest_position = self.positions[0].copy()
        self.gbest_fitness = -np.inf
        self.gbest_eta_air = 0.0
        self.gbest_eta_water = 0.0

        self.w, self.c1, self.c2 = 0.5, 1.5, 1.5

        # 并行计算配置 - 限制进程数避免资源占用过高
        self.num_processes = min(cpu_count(), 4)  # 最多4个进程，给系统留出余地
        print(f"🔄 将使用 {self.num_processes} 个进程并行评估（限制进程数以降低资源占用）")

        # 早停机制相关变量
        self.last_gbest_fitness = -np.inf
        self.no_improve_count = 0
        self.early_stop_generations = 20  # 连续20代无改进则停止

    def run(self):
        print(f"🚀 21维跨介质 PSO (厚度/光顺度双约束版) 启动...")
        print(f"⚙️  并行配置：{self.num_processes} 进程，{self.num_particles} 粒子，{self.max_iter} 代")

        for iter_num in range(self.max_iter):
            alive_count = 0

            # 1. 几何验证（串行，但很快）
            valid_indices = []
            params_to_evaluate = []

            for i in range(self.num_particles):
                if not self.validator.is_valid(self.positions[i][3:21]):
                    eta_air, eta_water = -1.0, -1.0
                else:
                    valid_indices.append(i)
                    params_to_evaluate.append(self.positions[i].tolist())

            # 2. 并行评估合法粒子（使用优化版本）
            if params_to_evaluate:
                try:
                    with Pool(processes=self.num_processes, initializer=init_worker) as pool:
                        results = pool.map(evaluate_particle_optimized, params_to_evaluate)

                    # 3. 写回结果
                    for idx, (eta_air, eta_water) in zip(valid_indices, results):
                        if eta_air <= 0 or eta_water <= 0 or np.isnan(eta_air) or np.isnan(eta_water):
                            eta_air, eta_water = -1.0, -1.0

                        alive_count += 1 if eta_air > 0 and eta_water > 0 else 0
                        fitness = self.w_air * eta_air + self.w_water * eta_water

                        if fitness > self.pbest_fitness[idx]:
                            self.pbest_fitness[idx] = fitness
                            self.pbest_positions[idx] = self.positions[idx].copy()

                        if fitness > self.gbest_fitness:
                            self.gbest_fitness = fitness
                            self.gbest_position = self.positions[idx].copy()
                            self.gbest_eta_air = eta_air
                            self.gbest_eta_water = eta_water
                except Exception as e:
                    print(f"⚠️  并行评估出错: {e}，回退到串行模式")
                    # 回退到串行模式
                    for idx, params in zip(valid_indices, params_to_evaluate):
                        try:
                            eta_air, eta_water = evaluate_individual(params)
                            if eta_air <= 0 or eta_water <= 0 or np.isnan(eta_air) or np.isnan(eta_water):
                                eta_air, eta_water = -1.0, -1.0

                            alive_count += 1 if eta_air > 0 and eta_water > 0 else 0
                            fitness = self.w_air * eta_air + self.w_water * eta_water

                            if fitness > self.pbest_fitness[idx]:
                                self.pbest_fitness[idx] = fitness
                                self.pbest_positions[idx] = self.positions[idx].copy()

                            if fitness > self.gbest_fitness:
                                self.gbest_fitness = fitness
                                self.gbest_position = self.positions[idx].copy()
                                self.gbest_eta_air = eta_air
                                self.gbest_eta_water = eta_water
                        except Exception as e2:
                            print(f"⚠️  串行评估也出错: {e2}")
                            eta_air, eta_water = -1.0, -1.0
                            fitness = self.w_air * eta_air + self.w_water * eta_water

                            if fitness > self.pbest_fitness[idx]:
                                self.pbest_fitness[idx] = fitness
                                self.pbest_positions[idx] = self.positions[idx].copy()

                            if fitness > self.gbest_fitness:
                                self.gbest_fitness = fitness
                                self.gbest_position = self.positions[idx].copy()
                                self.gbest_eta_air = eta_air
                                self.gbest_eta_water = eta_water

            # 4. 更新非法粒子的适应度
            invalid_indices = set(range(self.num_particles)) - set(valid_indices)
            for i in invalid_indices:
                eta_air, eta_water = -1.0, -1.0
                fitness = self.w_air * eta_air + self.w_water * eta_water

                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()

                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
                    self.gbest_eta_air = eta_air
                    self.gbest_eta_water = eta_water

            # 5. 打印输出（减少频率，每10代打印一次）
            if (iter_num + 1) % 10 == 0 or iter_num == self.max_iter - 1:
                print(f"========== 第 {iter_num + 1}/{self.max_iter} 代: 存活且合规个体 {alive_count}/{self.num_particles} ==========")
                if self.gbest_fitness != -np.inf:
                    print(f"当前全局最高加权得分: {self.gbest_fitness:.4f} (对应气动效率: {self.gbest_eta_air:.4f}, 水动效率: {self.gbest_eta_water:.4f})")

            # 6. 早停检查
            if iter_num > 0:
                improvement = self.gbest_fitness - self.last_gbest_fitness
                if abs(improvement) < 1e-6:
                    self.no_improve_count += 1
                else:
                    self.no_improve_count = 0
                    self.last_gbest_fitness = self.gbest_fitness

                if self.no_improve_count >= self.early_stop_generations:
                    print(f"⏹️  提前终止：连续{self.early_stop_generations}代无显著改进")
                    break
            else:
                self.last_gbest_fitness = self.gbest_fitness
                self.no_improve_count = 0

            # 7. 粒子位置更新
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
    # ===== 【固定随机种子，确保可复现性】 =====
    RANDOM_SEED = 42  # 可改为任意整数
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    base_airfoil_path = os.path.join(os.path.dirname(__file__), "..", "xfoil_runner", "CLARKY_geo.dat")
    base_airfoil_path = os.path.join(os.path.dirname(__file__), "..", "xfoil_runner", "CLARKY_geo.dat")
    
    if not os.path.exists(base_airfoil_path):
        print(f"❌ 找不到基础翼型文件: {base_airfoil_path}")
        print("请检查你的路径配置！")
        sys.exit()

    # 初始化几何宪兵
    geom_validator = GeometryValidator(base_airfoil_path)
    
    # 物理参数边界设定
    bounds_3 = [(0.15, 0.24), (2.0, 100.0), (0.3, 0.5)]  # cm, beta, m
    bounds_18 = [(-0.01, 0.01)] * 18  # 翼型权重，保持 ±0.01
    full_bounds = bounds_3 + bounds_18  # 总共21维

    # 启动 PSO（21维优化框架）
    pso = PSO(num_particles=200, num_vars=21, bounds=full_bounds, max_iter=100, validator=geom_validator)
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
        
        print(f"[最佳 21 维参数解]")
        print(f"三维弦长参数: cm = {best_position[0]:.4f}, beta = {best_position[1]:.4f}, m = {best_position[2]:.4f}")
        print("二维翼型权重 (18维):")
        print(np.round(best_position[3:21], 4))
        
        # 打印详细解剖数据
        print_optimal_details(best_position.tolist())