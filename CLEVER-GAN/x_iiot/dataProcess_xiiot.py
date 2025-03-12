from sklearn.preprocessing import LabelEncoder
import pandas as pd
df = pd.read_csv('/x_iiot_40000_10class.csv', encoding='utf-8', low_memory=False)
drop_col = ["Bad_checksum", "is_SYN_with_RST"]
df = df.drop(columns=drop_col)
cols = df.columns
for col in cols[0:-2]:
    if df[col].dtype == object or bool:
        encoder = LabelEncoder().fit(df[col])
        df[col] = encoder.transform(df[col])
ATTACK_CAT_TO_ID = {
    'Normal': 0,
    'C&C': 1,
    'Exfiltration': 2,
    'Exploitation': 3,
    'Lateral _movement': 4,
    'RDOS': 5,
    'Reconnaissance': 6,
    'Tampering': 7,
    'Weaponization': 8,
    'crypto-ransomware': 9,
}
ATTACK_TWO = {
    'Normal': 0,
    'Attack': 1,
}

df['class2'] = df['class2'].apply(func=(lambda x: ATTACK_CAT_TO_ID.get(x)))
df['class3'] = df['class3'].apply(func=(lambda x: ATTACK_TWO.get(x)))

# 数据特征归一化处理
result1 = df.copy()

# 正则化
result = result1.copy()
for feature_name in cols[0:-3]:
    mean_value = result1[feature_name].mean()
    std_value = result1[feature_name].std()
    result[feature_name] = (result1[feature_name] - mean_value) / std_value


result.to_csv('datasets_xiiot_afterProcess.csv', index=False)
