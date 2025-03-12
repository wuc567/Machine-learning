from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

datasets = pd.read_csv('D:/python/pythonProject/PHDfirstTest/Train_Test_Windows_10.csv', encoding='utf-8',
                       low_memory=False)

print(datasets.shape)

# 将空字符串 " " 替换为 NaN
datasets.replace(" ", np.nan, inplace=True)

# Remove infinite values
datasets.replace([-np.inf, np.inf], np.nan, inplace=True)  # 将极大极小值化为 NAN
datasets.dropna(axis=0, how='any', inplace=True)  # 删除缺失值（NAN）

# 查找全为零的列
zero_columns = datasets.columns[(datasets == 0).all()]
for i in zero_columns:
    datasets = datasets.drop(i, axis=1)

# 判断空值
q = np.any(pd.isnull(datasets))
print(q)




ATTACK_CAT_TO_ID = {
    'dos': 0,
    'ddos': 1,
    'injection': 2,
    'normal': 3,
    'xss': 4,
    'password': 5,
    'scanning': 6,
    'mitm': 7,
}

datasets['type'] = datasets['type'].apply(func=(lambda x: ATTACK_CAT_TO_ID.get(x)))
datasets = datasets.drop(columns='label')

print(datasets.shape)

# 删除包含非数字值的行
datasets = datasets.apply(pd.to_numeric, errors='coerce').dropna()

print(datasets.shape)

# 查找无法转换为数字的值
for col in datasets.columns:
    # 使用 try-except 来捕获错误
    try:
        datasets[col].astype(float)
    except ValueError:
        print(f"Column {col} contains non-numeric values.")

# 数据特征归一化处理
result1 = datasets.copy()
cols = datasets.columns

# 正则化
result = result1.copy()
for feature_name in cols[0:-1]:
    mean_value = result1[feature_name].mean()
    std_value = result1[feature_name].std()
    result[feature_name] = (result1[feature_name] - mean_value) / std_value

print(result.shape)

# 判断空值
q = np.any(pd.isnull(result))
print(q)
result.dropna(axis=1, how='any', inplace=True)  # 删除缺失值（NAN）

print(result.shape)

q = np.any(pd.isnull(result))
print(q)
result.to_csv('datasets_ton_iot_afterProcess.csv', index=False)