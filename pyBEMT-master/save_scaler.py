# 建议新建一个名为 save_scaler.py 的独立脚本运行一次即可
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# 1. 读取一次庞大的 CSV
print("正在读取全量数据库...")
df = pd.read_csv('Airfoil_Aerodynamic_Database.csv')

# 2. 划定列名
feature_cols = [f'w{i}' for i in range(1, 19)] + ['Re', 'alpha']
target_cols = ['CL', 'CD']

# 3. 拟合尺子
print("正在拟合 StandardScaler...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(df[feature_cols].values)
scaler_y.fit(df[target_cols].values)

# 4. 保存为 pkl 文件
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✅ Scaler 已经成功保存为 scaler_X.pkl 和 scaler_y.pkl！以后无需再读取 CSV！")