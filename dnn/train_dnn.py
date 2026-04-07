import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# ==========================================
# 1. 读取数据
# ==========================================
print("正在加载气动数据库...")
df = pd.read_csv('Airfoil_Aerodynamic_Database.csv') # 确保文件名和你的真实文件一致

feature_cols = [f'w{i}' for i in range(1, 19)] + ['Re', 'alpha']
target_cols = ['CL', 'CD']

X = df[feature_cols].values 
y = df[target_cols].values  

print(f"输入数据维度: {X.shape}, 输出数据维度: {y.shape}")

# ==========================================
# 2. 数据预处理 (洗菜与切配)
# ==========================================
print("正在进行数据标准化与张量转换...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

# ==========================================
# 3. 搭建深度全连接神经网络 (DNN)
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

model = AirfoilDNN()

# ==========================================
# 4. 设置损失函数与优化器
# ==========================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 5. 启动训练循环 (炼丹)
# ==========================================
epochs = 1000  # 数据量大了，我们让它多学一会儿，设为1000轮
print("\n--- 开始训练网络 (由于数据有20万行，请耐心等待) ---")

start_time = time.time()

for epoch in range(epochs):
    model.train()
    
    # 前向传播
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    
    # 反向传播
    optimizer.zero_grad() 
    loss.backward()       
    optimizer.step()      
    
    # 每 100 轮汇报一次成绩
    if (epoch + 1) % 100 == 0:
        model.eval() 
        with torch.no_grad(): 
            test_preds = model(X_test_tensor)
            test_loss = criterion(test_preds, y_test_tensor)
        print(f"Epoch [{epoch+1}/{epochs}] | 训练集 Loss: {loss.item():.4f} | 测试集 Loss: {test_loss.item():.4f}")

end_time = time.time()
print(f"\n模型训练完毕！总耗时: {(end_time - start_time):.2f} 秒")

# 保存模型权重
torch.save(model.state_dict(), 'airfoil_dnn_weights.pth')
print("✅ 模型参数已安全保存为 'airfoil_dnn_weights.pth'")