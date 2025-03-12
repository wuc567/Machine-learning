import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import linalg
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity

def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也要设置随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内置随机数生成器的随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作的结果确定
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化算法选择


set_seed(42)  # 设置为固定的随机种子，例如42
# 生成新数据
def generate_data(model, num_samples, latent_dim):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        generated_data = model.vae.decoder(z)
    return generated_data


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

#
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/little_class_data.csv', encoding='utf-8',
#                         low_memory=False)
df = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/datasets_ton_iot_afterProcess.csv', encoding='utf-8',
                 low_memory=False)

data = df.to_numpy()

feature = data[:, 0:-1]
lable = data[:, -1]

print(feature.shape)
print(lable.shape)

# 你要根据第二列的值来筛选行
value_to_find = 3

# 使用布尔索引筛选行
filtered_rows = data[data[:, -1] == value_to_find]

real_data = filtered_rows[:, 0:-1]
real_label = filtered_rows[:, -1]

# # 把最后一个小样本类数据舍弃掉，目的是凑成能整除的数
# real_data = real_data[0:-1, :]
# real_label = real_label[0:-1]

# real_data就是真实的小样本数据
print(real_data)
print(real_data.shape)

data_final = real_data

# np.savetxt('3_class_data.csv', data_final, delimiter=',')

# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
# # 定义 VAE 模型
# class VAE(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(VAE, self).__init__()
#         # 编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, latent_dim * 2)  # 输出均值和方差的对数
#         )
#
#         # 解码器
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, input_dim),
#             # nn.ReLU()  # 输出值在[0,1]之间，视数据而定可换为tanh或其他激活函数
#         )
#
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         # 编码
#         encoded = self.encoder(x)
#         mu, log_var = torch.chunk(encoded, 2, dim=-1)  # 将输出拆分为均值和方差
#         z = self.reparameterize(mu, log_var)  # 重新参数化采样
#         # 解码
#         decoded = self.decoder(z)
#         return decoded, mu, log_var
#
#
# # 定义损失函数
# def loss_function(recon_x, x, mu, log_var):
#     # BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#     # 如果数据不在 [0, 1] 之间，使用 MSE
#     MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#     # return BCE + KLD
#     return MSE + KLD
#
#
# # 训练 VAE
# def train_vae(model, data, epochs=2000, batch_size=4, learning_rate=0.001):
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     fid_min = float('inf')
#     avg_distance = 0.0
#     average_cosine_similarity = 0.0
#     model.train()
#
#     for epoch in range(epochs):
#         for i in range(0, len(data), batch_size):
#             batch = data[i:i + batch_size]
#             optimizer.zero_grad()
#             recon_batch, mu, log_var = model(batch)
#             # print("recon_batch.shape:", recon_batch.shape)
#             # print("batch.shape:", batch.shape)
#             loss = loss_function(recon_batch, batch, mu, log_var)
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
#         # -------------------------------------------------------------------------------------------------------------------------------------------
#         # -------------------------------------------------------------------------------------------------------------------------------------------
#         # -------------------------------------------------------------------------------------------------------------------------------------------
#         # -------------------------------------------------------------------------------------------------------------------------------------------
#         # 生成1000条新数据
#         num_samples = 13
#         generated_data = generate_data(model, num_samples=num_samples, latent_dim=1)
#
#         # 转换为numpy数组并保存为csv文件
#         generated_data_numpy = generated_data.numpy()
#
#         # np.savetxt('vae_gan_contra_generated_data.csv', generated_data_numpy, delimiter=',')
#         np.savetxt('vae_generated_data.csv', generated_data_numpy, delimiter=',')
#         data_generate = pd.read_csv(
#             'D:/python/pythonProject/PHDfirstTest/ton_iot/vae_generated_data.csv', encoding='utf-8',
#             low_memory=False, header=None, dtype='float64')
#         # 计算真实样本的均值和协方差
#         mu_real, sigma_real = calculate_statistics(data_text)
#
#         # 计算生成样本的均值和协方差
#         mu_gen, sigma_gen = calculate_statistics(data_generate)
#
#         # 计算 FID
#         fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
#         if fid_value < fid_min:
#             fid_min = fid_value
#             # 计算生成样本之间的欧氏距离
#             distances = pdist(data_generate, metric='euclidean')
#             avg_distance = np.mean(distances)
#             # 计算样本之间的余弦相似度
#             cosine_sim_matrix = cosine_similarity(data_generate)
#             # 去除对角线元素（每个样本与自己比较的相似度为 1），只保留两两不同样本之间的相似度
#             cosine_sim_matrix_no_diag = cosine_sim_matrix - np.eye(cosine_sim_matrix.shape[0])
#             # 将矩阵合并为一个值：取所有相似度值的平均值
#             average_cosine_similarity = np.mean(cosine_sim_matrix_no_diag[cosine_sim_matrix_no_diag != 0])
#
#             np.savetxt('best_vae_generated_data.csv', generated_data_numpy, delimiter=',')
#
#             # # 保存模型，保存模型的状态字典
#             # torch.save(vae_gan.state_dict(),
#             #            'D:/python/pythonProject/PHDfirstTest/UNSW/vae_gan_contra_lossMethod_generated_best_model.pth')
#     return fid_min, avg_distance, average_cosine_similarity
#
#
# # 生成新数据
# def generate_data(model, num_samples, latent_dim):
#     model.eval()
#     with torch.no_grad():
#         z = torch.randn(num_samples, latent_dim)
#         generated_data = model.decoder(z)
#     return generated_data
#
#
# # 数据预处理（假设数据已经加载到变量 `data` 中）
# # data = torch.tensor(your_data, dtype=torch.float32)
# data_final = torch.tensor(data_final, dtype=torch.float32)
#
# input_dim = 105
# latent_dim = 1  # 隐空间维度，可根据需要调整
#
# # 初始化模型
# vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
#
# # 训练模型
# fid_min, avg_distance, average_cosine_similarity = train_vae(vae, data_final)
# print(f"最佳FID between real and generated samples：{fid_min}")
# print(f"最佳Average distance between generated samples：{avg_distance}")
# print(f"最佳生成样本之间的平均余弦相似度：{average_cosine_similarity}")
#
# # # 生成1000条新数据
# # num_samples = 13
# # generated_data = generate_data(vae, num_samples=num_samples, latent_dim=latent_dim)
# #
# # # 转换为numpy数组并保存为csv文件
# # generated_data_numpy = generated_data.numpy()
# #
# #
# # np.savetxt('vae_generated_data.csv', generated_data_numpy, delimiter=',')
#
#
#
#
#
#
#
#
