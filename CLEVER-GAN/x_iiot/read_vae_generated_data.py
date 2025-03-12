import pandas as pd
df = pd.read_csv('D:/python/pythonProject/PHDfirstTest/generated_data.csv', encoding='utf-8',
                 low_memory=False)

data = df.to_numpy()


print(data.shape)