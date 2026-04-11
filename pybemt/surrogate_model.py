import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 定义网络结构 (必须和训练时一模一样)
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

# 2. 封装预测器类
class AirfoilPredictor:
    def __init__(self, csv_path='Airfoil_Aerodynamic_Database.csv', weights_path='airfoil_dnn_weights.pth'):
        """初始化阶段：只在程序刚启动时执行一次，用于加载脑图和刻度尺"""
        print("正在初始化 AI 代理模型，请稍候...")
        
        # 加载数据以校准翻译器
        df = pd.read_csv(csv_path)
        feature_cols = [f'w{i}' for i in range(1, 19)] + ['Re', 'alpha']
        target_cols = ['CL', 'CD']
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_X.fit(df[feature_cols].values)
        self.scaler_y.fit(df[target_cols].values)
        
        del df # 释放内存
        
        # 唤醒神经网络
        self.model = AirfoilDNN()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval() # 锁定为预测模式
        print("✅ AI 预测器已准备就绪！")

    def predict(self, weights_list, Re, alpha):
        """
        调用阶段：输入 18个权重、雷诺数、迎角，瞬间返回 CL 和 CD
        """
        # 组合这 20 个数字
        input_data = np.array(weights_list + [Re, alpha]).reshape(1, -1)
        
        # 1. 压缩输入
        input_scaled = self.scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)
        
        # 2. 网络瞬间预测
        with torch.no_grad():
            output_scaled = self.model(input_tensor).numpy()
            
        # 3. 翻译输出
        output_real = self.scaler_y.inverse_transform(output_scaled)
        
        CL = output_real[0][0]
        CD = output_real[0][1]
        
        return CL, CD

