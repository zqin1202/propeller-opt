import numpy as np
import pandas as pd
import subprocess
import os
import time
import datetime

# ==========================================
# 模块1：几何变形器 (自适应格式 + Improved HH)
# ==========================================
def generate_airfoil(base_dat_path, weights, output_path):
    coords = []
    with open(base_dat_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
    coords = np.array(coords)
    x_base = coords[:, 0]
    y_base = coords[:, 1]

    # 自适应数据切分 (兼容 Selig 与 Lednicer 格式)
    diffs = np.diff(x_base)
    split_indices = np.where(diffs < -0.5)[0]

    if len(split_indices) > 0:
        split_idx = split_indices[0] + 1
        x_1_raw, y_1_raw = x_base[:split_idx], y_base[:split_idx]
        x_2_raw, y_2_raw = x_base[split_idx:], y_base[split_idx:]
    else:
        le_idx = np.argmin(x_base)
        x_1_raw, y_1_raw = x_base[:le_idx+1], y_base[:le_idx+1]
        x_2_raw, y_2_raw = x_base[le_idx:], y_base[le_idx:]

    # 强制递增排序以适配插值
    if x_1_raw[0] > x_1_raw[-1]: x_1_raw, y_1_raw = x_1_raw[::-1], y_1_raw[::-1]
    if x_2_raw[0] > x_2_raw[-1]: x_2_raw, y_2_raw = x_2_raw[::-1], y_2_raw[::-1]

    beta = np.linspace(0, np.pi, 100)
    x_std = 0.5 * (1 - np.cos(beta))
    y_1_std = np.interp(x_std, x_1_raw, y_1_raw)
    y_2_std = np.interp(x_std, x_2_raw, y_2_raw)

    # 智能物理面识别
    mid_idx = len(x_std) // 2
    if y_1_std[mid_idx] > y_2_std[mid_idx]:
        y_upper_std, y_lower_std = y_1_std, y_2_std
    else:
        y_upper_std, y_lower_std = y_2_std, y_1_std

    y_upper_new = np.copy(y_upper_std)
    y_lower_new = np.copy(y_lower_std)

    # 改进型 Hicks-Henne 变形
    peak_x_mid = np.array([0.10, 0.23, 0.37, 0.50, 0.63, 0.77, 0.90])
    A_TE, B_TE = 1.0, 15.0

    dy_upper = np.zeros_like(x_std)
    dy_lower = np.zeros_like(x_std)

    for i in range(9): 
        if i == 0:
            bump = (x_std**0.5) * (1 - x_std) * np.exp(-15 * x_std)
        elif i == 8:
            bump = A_TE * (x_std**0.5) * (1 - x_std) * np.exp(-B_TE * (1 - x_std))
        else:
            x_c = peak_x_mid[i-1] 
            e_i = np.log(0.5) / np.log(x_c)
            bump = (np.sin(np.pi * x_std**e_i))**3
            
        dy_upper += weights[i] * bump
        dy_lower += weights[i+9] * bump
        
    y_upper_new += dy_upper
    y_lower_new += dy_lower

    # 几何交叉拦截器 (保留此项以防 XFOIL 卡死)
    if np.any(y_upper_new[5:-5] <= y_lower_new[5:-5]):
        return False

    # 拼接并输出标准 Selig 格式
    x_final = np.concatenate((x_std[::-1], x_std[1:]))
    y_final = np.concatenate((y_upper_new[::-1], y_lower_new[1:]))

    with open(output_path, 'w') as f:
        f.write("Generated_Airfoil\n")
        for i in range(len(x_final)):
            f.write(f"{x_final[i]:.6f}  {y_final[i]:.6f}\n")
            
    return True

# ==========================================
# 模块2：XFOIL 自动化引擎 (静默执行与动态清理)
# ==========================================
def run_xfoil(dat_filename, Re, alpha_start=0, alpha_end=10, alpha_step=1, file_id="0"):
    polar_file = f"tmp_pol_{file_id}.txt"
    if os.path.exists(polar_file):
        os.remove(polar_file)

    input_commands = (
        f"load {dat_filename}\n"  
        "pane\n"                  
        "oper\n"                  
        f"visc {Re}\n"            
        "iter 200\n"              
        "pacc\n"                  
        f"{polar_file}\n"         
        "\n"                      
        f"aseq {alpha_start} {alpha_end} {alpha_step}\n"
        "\n"                      
        "quit\n"                  
    )

    try:
        subprocess.run("xfoil.exe", input=input_commands, capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None

    clean_df = None
    if os.path.exists(polar_file):
        try:
            df = pd.read_csv(polar_file, skiprows=12, sep='\s+', names=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'])
            clean_df = df[(df['CD'] > 0.001) & (df['CD'] < 0.2) & (df['CL'] > -0.5) & (df['CL'] < 1.6)].copy()
            clean_df = clean_df[['alpha', 'CL', 'CD']]
        except Exception:
            pass
        finally:
            try:
                os.remove(polar_file)
            except OSError:
                pass 

    return clean_df

# ==========================================
# 模块3：主程序 - 静默量产车间 (带独立版本文件夹)
# ==========================================
if __name__ == "__main__":
    base_file = "CLARKY_geo.dat" 
    N_samples = 10000  # 计划生成的总样本数
    
    # 💡 1. 生成带有时间戳和样本数的独立文件夹名称
    # 格式例如: Database_20260316_153022_N10000
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"Database_{current_time}_N{N_samples}"
    
    # 💡 2. 创建该文件夹 (如果不存在的话)
    os.makedirs(output_folder, exist_ok=True)
    
    # 💡 3. 将数据库文件路径指向该文件夹内部
    database_file = os.path.join(output_folder, "Airfoil_Aerodynamic_Database.csv")
    
    print(f"--- 启动 XFOIL 数据量产工厂 (静默模式) ---")
    print(f"计划生成样本数: {N_samples}")
    print(f"📁 本次运行数据将安全隔离保存在: {output_folder}/")
    print("运行中，将每隔 100 个样本汇报一次进度...\n")
    
    write_header = not os.path.exists(database_file)
    start_time = time.time()
    
    success_count = 0
    discard_count = 0

    for i in range(N_samples):
        weights = np.random.uniform(-0.015, 0.015, 18)
        geo_filename = f"tmp_geo_{i}.dat"
        
        is_geometry_valid = generate_airfoil(base_file, weights, geo_filename)
        
        if not is_geometry_valid:
            discard_count += 1
            if os.path.exists(geo_filename): os.remove(geo_filename)
            continue  
            
        for Re in [50000, 200000]:
            polar_df = run_xfoil(geo_filename, Re, alpha_start=0, alpha_end=10, alpha_step=1, file_id=str(i))
            
            if polar_df is not None and not polar_df.empty:
                for j in range(18):
                    polar_df.insert(j, f'w{j+1}', weights[j])
                polar_df.insert(18, 'Re', Re)
                
                polar_df.to_csv(database_file, mode='a', header=write_header, index=False)
                write_header = False  
                success_count += len(polar_df)

        try:
            if os.path.exists(geo_filename):
                os.remove(geo_filename)
        except OSError:
            pass

        # 里程碑进度汇报
        if (i + 1) % 100 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"[{i+1}/{N_samples}] 已耗时 {elapsed:.1f} 分钟 | 累计入库: {success_count} 行有效数据 | 累计拦截废弃几何: {discard_count} 个")

    end_time = time.time()
    print("\n" + "="*50)
    print("✅ 数据库生成完毕！")
    print(f"总耗时: {(end_time - start_time) / 3600:.2f} 小时")
    print(f"成功采集到 {success_count} 条有效气动数据行！")
    print(f"总拦截废弃几何数: {discard_count}")
    print(f"📁 数据已安全存入: {database_file}")
    print("="*50)