import os
import subprocess
import pandas as pd

def run_xfoil_naca(naca_code, Re, alpha_start, alpha_end, alpha_step):
    """
    自动化调用 XFOIL 计算 NACA 翼型的气动数据
    """
    # 1. 定义输出文件的名称
    polar_file = f"polar_NACA{naca_code}_Re{Re}.txt"
    
    # 如果之前有同名旧文件，先删掉，防止数据追加混淆
    if os.path.exists(polar_file):
        os.remove(polar_file)

    # 2. 编写 XFOIL 控制指令串 (这正是你手动操作 XFOIL 时敲的那些命令)
    # 注意 \n 代表模拟键盘按下回车键
    input_commands = (
        f"naca {naca_code}\n"   # 生成 NACA 翼型
        "pane\n"                # 平滑面板节点 (消除尖锐点)
        "oper\n"                # 进入操作(Operation)模式
        f"visc {Re}\n"          # 开启粘性模式并设置雷诺数
        "iter 200\n"            # 把最大迭代次数设置到200次，防止难收敛的工况报错
        "pacc\n"                # 开启极曲线自动保存(Polar Accumulation)
        f"{polar_file}\n"       # 输入保存数据的文件名
        "\n"                    # 倾倒文件(Dump file)，我们不需要，直接敲回车跳过
        f"aseq {alpha_start} {alpha_end} {alpha_step}\n"  # 执行攻角扫描 (起始 终止 步长)
        "\n"                    # 结束计算，退回主菜单
        "quit\n"                # 退出 XFOIL
    )

    print(f"开始后台计算: NACA {naca_code}, Re={Re}...")

    # 3. 核心步骤：使用 subprocess 在后台唤醒 xfoil.exe 并输入指令
    # 要求 xfoil.exe 必须和本 Python 脚本在同一个文件夹下
    try:
        process = subprocess.Popen(
            "xfoil.exe",               # 调用的程序名
            stdin=subprocess.PIPE,     # 允许我们通过管道向它发送键盘输入
            stdout=subprocess.PIPE,    # 捕获它的屏幕输出（防止弹出一堆黑框）
            stderr=subprocess.PIPE,    # 捕获报错
            text=True                  # 以字符串形式交互
        )
        
        # 将我们准备好的指令串发送给 XFOIL，并等待它执行完毕
        stdout, stderr = process.communicate(input_commands)
        print("XFOIL 计算完成！")
        
    except FileNotFoundError:
        print("错误：找不到 xfoil.exe！请确保它和这个 Python 脚本在同一个文件夹下。")
        return None

    # 4. 利用 Pandas 读取生成的 txt 文件中的升阻力数据
    if os.path.exists(polar_file):
        # XFOIL 的输出文件前 12 行是表头说明，我们需要跳过 (skiprows)
        # 它的数据是用不规则空格分隔的，所以用到正则分隔符 '\s+'
        try:
            df = pd.read_csv(polar_file, skiprows=12, sep='\s+', 
                             names=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'])
            print("\n提取到的核心数据如下：")
            print(df[['alpha', 'CL', 'CD']]) # 只打印我们最关心的攻角、升力、阻力
            return df
        except Exception as e:
            print(f"读取数据时出错：{e}")
            return None
    else:
        print("警告：XFOIL 未能成功生成结果文件，可能是计算发散了。")
        return None

# ==========================================
# 主程序测试区
# ==========================================
if __name__ == "__main__":
    # 按照我们之前讨论的降维策略：选用空中典型雷诺数 50000，攻角 0 到 10 度，步长 1 度
    result_df = run_xfoil_naca(naca_code="4412", Re=500000, alpha_start=0, alpha_end=10, alpha_step=1)
    
    # 你可以把结果保存为更容易处理的 csv 格式，方便以后喂给神经网络
    if result_df is not None:
        result_df.to_csv("training_data_sample.csv", index=False)
        print("\n数据已保存为 training_data_sample.csv ！")