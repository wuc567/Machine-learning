import numpy as np
import torch
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.12f' % x)  # 设置浮点数显示精度为12位

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/generated_data.csv', encoding='utf-8',
#                             low_memory=False, header=None, dtype='float64')

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/vae_gan_contra_lossMethod_generated_data.csv', encoding='utf-8',
#                             low_memory=False, header=None, dtype='float64')

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/vae_gan_contra_generated_data.csv', encoding='utf-8',
#                             low_memory=False, header=None, dtype='float64')


data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/GAN_generated_data.csv', encoding='utf-8',
                            low_memory=False, header=None, dtype='float64')

data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/datasets_xiiot_afterProcess.csv', encoding='utf-8',
                        low_memory=False)

print(data_generate)

data_text = data_text.to_numpy()
data_generate = data_generate.to_numpy()

# 假设我们要获取倒数第二列中的值等于 9 的所有行
target_value = 9

# 使用布尔索引获取倒数第二列等于 target_value 的行
text_Result = data_text[data_text[:, -2] == target_value]

# # 使用 numpy.delete() 删除倒数第一列和倒数第三列
# text_Result = np.delete(text_Result, [index_third_last, index_last], axis=1)
text_Result = text_Result[:, :-3]

# 假设 gen_samples 是生成的 1000 个样本，shape 是 [1000, 57]
# 假设 real_samples 是 21 条真实样本，shape 是 [21, 57]

gen_samples = data_generate  # 生成的样本
real_samples = text_Result  # 真实样本

# print(real_samples)


np.savetxt('real_little_class_data.csv', real_samples, delimiter=',')












gen_samples = torch.tensor(gen_samples, dtype=torch.float32)
real_samples = torch.tensor(real_samples, dtype=torch.float32)

print(gen_samples.shape)
print(real_samples.shape)

# 计算每个生成样本与所有真实样本的差距并求平均
total_diff = 0
diffs_array = []
diffs_temp = 0

len_gen = gen_samples.shape[0]
len_real = real_samples.shape[0]

for gen_sample in gen_samples:
    for real_sample in real_samples:
        # print(gen_sample - real_sample)
        diffs_temp = diffs_temp + torch.abs(gen_sample - real_sample).sum(dim=0)  # 计算每个生成样本与每个真实样本的差距 (L1 距离)
        # print("diffs_temp1:", diffs_temp)

    diffs_temp = diffs_temp / len_real
    diffs_array.append(diffs_temp)
    # print("diffs_temp2:", diffs_temp)
    diffs_temp = 0

for i in diffs_array:
    total_diff = total_diff + i

print(f"生成样本与真实样本的总差距: {total_diff.item()}")
