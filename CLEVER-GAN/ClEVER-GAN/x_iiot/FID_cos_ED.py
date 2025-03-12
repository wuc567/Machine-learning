import numpy as np
from scipy import linalg
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.12f' % x)  # 设置浮点数显示精度为12位

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/vae_generated_data.csv', encoding='utf-8',
#                             low_memory=False, header=None, dtype='float64')

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/vae_gan_contra_lossMethod_generated_data.csv', encoding='utf-8',
#                             low_memory=False, header=None, dtype='float64')


# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/vae_gan_generated_data.csv', encoding='utf-8',
#                             low_memory=False, header=None, dtype='float64')

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/vae_gan_contra_generated_data.csv', encoding='utf-8',
#                             low_memory=False, header=None, dtype='float64')


data_generate = pd.read_csv('/GAN_generated_data.csv', encoding='utf-8',
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

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    计算 Fréchet Distance.
    参数:
    - mu1: 真实数据均值 (57维向量)
    - sigma1: 真实数据协方差矩阵 (57x57)
    - mu2: 生成数据均值 (57维向量)
    - sigma2: 生成数据协方差矩阵 (57x57)
    - eps: 一个小值，用于数值稳定性

    返回值:
    - Fréchet Distance (FID)
    """
    # 计算均值差的平方范数
    diff = mu1 - mu2
    # 计算协方差矩阵的平方根
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # 如果协方差矩阵不正定，可能会出现复数，需要去掉复数部分
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算 Fréchet Distance
    fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def calculate_statistics(data):
    """
    计算数据的均值和协方差
    参数:
    - data: 数据矩阵，形状为 (n_samples, 57)

    返回值:
    - mu: 数据均值 (57维向量)
    - sigma: 数据协方差矩阵 (57x57)
    """
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    return mu, sigma


# 真实样本 (21, 57) 和生成样本 (21, 57)
# 你需要替换成自己的真实数据和生成数据


# 计算真实样本的均值和协方差
mu_real, sigma_real = calculate_statistics(real_samples)

# 计算生成样本的均值和协方差
mu_gen, sigma_gen = calculate_statistics(gen_samples)

# 计算 FID
fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
print(f"FID between real and generated samples: {fid_value}")



from scipy.spatial.distance import pdist

# 计算生成样本之间的欧氏距离
distances = pdist(gen_samples, metric='euclidean')
avg_distance = np.mean(distances)
print(f"Average distance between generated samples: {avg_distance}")




import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 计算样本之间的余弦相似度
cosine_sim_matrix = cosine_similarity(gen_samples)

# # 输出余弦相似度矩阵
# print("余弦相似度矩阵：")
# print(cosine_sim_matrix)

# 去除对角线元素（每个样本与自己比较的相似度为 1），只保留两两不同样本之间的相似度
cosine_sim_matrix_no_diag = cosine_sim_matrix - np.eye(cosine_sim_matrix.shape[0])

# 将矩阵合并为一个值：取所有相似度值的平均值
average_cosine_similarity = np.mean(cosine_sim_matrix_no_diag[cosine_sim_matrix_no_diag != 0])

# 输出平均余弦相似度
print(f"生成样本之间的平均余弦相似度：{average_cosine_similarity:.4f}")