import torch
import torch.nn as nn
import numpy as np
import joblib

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
    def __init__(self, scaler_X_path='scaler_X.pkl', scaler_y_path='scaler_y.pkl', weights_path='airfoil_dnn_weights.pth'):
        # 检查是否在多进程环境下，如果是则抑制打印
        import sys
        import multiprocessing
        is_in_worker = hasattr(multiprocessing, 'current_process') and multiprocessing.current_process().name != 'MainProcess'

        if not is_in_worker:
            print("正在初始化 AI 代理模型，加载极速版...")

        # 设置确定性模式，确保可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 1. 直接一秒加载保存好的 scaler，不再读取 csv
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)

        # 2. 唤醒神经网络
        self.model = AirfoilDNN()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

        if not is_in_worker:
            print("✅ AI 预测器已极速准备就绪！")

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

