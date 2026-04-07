import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 重新构建相同的神经网络结构
# ==========================================
class AirfoilDNN(nn.Module):
    def __init__(self):
        super(AirfoilDNN, self).__init__()
        self.layer1 = nn.Linear(20, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.output_layer(x)
        return x

# ==========================================
# 2. 核心操作：用【原训练数据】装载和校准“翻译器”
# ==========================================
print("正在读取原训练数据以校准数据标准化比例尺...")
df_train = pd.read_csv('Airfoil_Aerodynamic_Database.csv') # 20万行的那个文件
feature_cols = [f'w{i}' for i in range(1, 19)] + ['Re', 'alpha']
target_cols = ['CL', 'CD']

scaler_X = StandardScaler()
scaler_y = StandardScaler()
# 注意：这里只用 df_train 进行 fit (校准刻度)
scaler_X.fit(df_train[feature_cols].values)
scaler_y.fit(df_train[target_cols].values)
print("翻译器校准完毕。\n")

# 清理内存，释放原训练数据占据的空间
del df_train 

# ==========================================
# 3. 唤醒保存的 AI 代理模型
# ==========================================
model = AirfoilDNN()
# 加载你之前跑了285秒保存下来的权重“记忆”
model.load_state_dict(torch.load('airfoil_dnn_weights.pth'))
model.eval() # 务必切换到测试评估模式！
print("✅ AI 代理模型已成功唤醒！\n")

# ==========================================
# 4. 读取全新的【独立测试集】并进行预测
# ==========================================
print("正在导入独立测试集进行盲测...")
df_test = pd.read_csv('Airfoil_Aerodynamic_Database_test.csv') # 123行的那个文件

test_X_raw = df_test[feature_cols].values
test_y_real = df_test[target_cols].values

# 用刚才校准好的翻译器，对测试集特征进行转换
test_X_scaled = scaler_X.transform(test_X_raw)
test_X_tensor = torch.FloatTensor(test_X_scaled)

# 代理模型瞬间完成 123 组工况的预测
with torch.no_grad():
    pred_y_scaled = model(test_X_tensor).numpy()

# 将预测出的标准化数据，翻译回真实的物理 CL 和 CD
pred_y_real = scaler_y.inverse_transform(pred_y_scaled)

# ==========================================
# 5. 深度误差分析与统计计算
# ==========================================
# 计算绝对误差 (|真实值 - 预测值|)
errors = np.abs(test_y_real - pred_y_real)
error_CL = errors[:, 0]
error_CD = errors[:, 1]

# 计算平均绝对误差 (MAE - Mean Absolute Error)
mae_CL = np.mean(error_CL)
mae_CD = np.mean(error_CD)

# 计算最大绝对误差 (Max Error)
max_err_CL = np.max(error_CL)
max_err_CD = np.max(error_CD)

print("================ 独立测试集验证报告 ================")
print(f"参与盲测的样本总数: {len(df_test)} 个\n")

print(f"【升力系数 CL 预测精度】")
print(f"  -> 平均绝对误差 (MAE): {mae_CL:.5f}")
print(f"  -> 最大绝对误差 (Max): {max_err_CL:.5f}\n")

print(f"【阻力系数 CD 预测精度】")
print(f"  -> 平均绝对误差 (MAE): {mae_CD:.5f}")
print(f"  -> 最大绝对误差 (Max): {max_err_CD:.5f}\n")

print("================ 随机抽取 5 组数据进行明细对比 ================")
sample_indices = np.random.choice(len(df_test), 5, replace=False)
for idx in sample_indices:
    Re_val = int(test_X_raw[idx][-2])
    alpha_val = test_X_raw[idx][-1]
    
    print(f"样本 #{idx+1} (Re={Re_val}, alpha={alpha_val}°):")
    print(f"  [XFOIL 真值] CL: {test_y_real[idx][0]:.4f}, CD: {test_y_real[idx][1]:.4f}")
    print(f"  [DNN   预测] CL: {pred_y_real[idx][0]:.4f}, CD: {pred_y_real[idx][1]:.4f}")
    print(f"  [绝对 误差 ] CL相差 {error_CL[idx]:.4f}, CD相差 {error_CD[idx]:.4f}")
    print("-" * 55)