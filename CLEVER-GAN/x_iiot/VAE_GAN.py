# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import torch
import torch.nn.functional as F

import random
import numpy as np

import torch.nn as nn
import torch.optim as optim


def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也要设置随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内置随机数生成器的随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作的结果确定
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化算法选择


set_seed(42)  # 设置为固定的随机种子，例如42
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


# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/little_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/0_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/1_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/4_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/5_class_data.csv', encoding='utf-8',
#                         low_memory=False)
data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/6_class_data.csv', encoding='utf-8',
                        low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/7_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/8_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/3_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# --------------------------------------------------------------准备正样本数据--------------------------------------------------------------------------------

df = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/datasets_xiiot_afterProcess.csv', encoding='utf-8',
                 low_memory=False)

data = df.to_numpy()
feature = data[:, 0:-3]
lable = data[:, -2]

print(lable.shape)

# ------------------------------------------------------------------------------------------------------------------------------------
# 你要根据第二列的值来筛选行
value_to_find = 6
# ------------------------------------------------------------------------------------------------------------------------------------


# 使用布尔索引筛选行
filtered_rows = data[data[:, -2] == value_to_find]

data_final = filtered_rows[:, 0:-3]

print(data_final)
print(data_final.shape)

# --------------------------------------------------------------准备负样本数据--------------------------------------------------------------------------------

data_fu = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/datasets_xiiot_afterProcess.csv', encoding='utf-8',
                      low_memory=False)

data_fu = data_fu.to_numpy()

# 指定要检查的列索引（从0开始）和要删除的值
column_index = -2  # 第2列（索引为1）

# -------------------------------------------------------------------------------------------------------
value_to_remove = 0
# -------------------------------------------------------------------------------------------------------


# 使用布尔索引来筛选出不包含指定值的行
# filtered_data = data_fu[data_fu[:, column_index] != value_to_remove]
filtered_data = data_fu[data_fu[:, column_index] == 0]

# 计算倒数第一列和倒数第三列的索引
index_last = -1  # 倒数第一列
index_third_last = -3  # 倒数第三列

# 使用 numpy.delete() 删除倒数第一列和倒数第三列
data_fu = np.delete(filtered_data, [index_third_last, index_last], axis=1)

# data_fu_x = data_fu[:39963, :-1]
# data_fu_y = data_fu[:39963, -1]

# print(data_fu_x)
# print(data_fu_x.shape)
# print(data_fu_y)
# print(data_fu_y.shape)

input_dim = 57
hidden_dim = 20
lr = 0.001
epochs = 5000
temperature = 0.2


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # 输出均值和方差的对数
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            # nn.ReLU()  # 输出值在[0,1]之间，视数据而定可换为tanh或其他激活函数
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)  # 将输出拆分为均值和方差
        z = self.reparameterize(mu, log_var)  # 重新参数化采样
        # 解码
        decoded = self.decoder(z)
        return decoded, mu, log_var


# 定义判别器 (Discriminator)
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(57, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# 定义 VAE-GAN 模型
class VAE_GAN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VAE_GAN, self).__init__()
        self.vae = VAE(input_dim, hidden_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim)

    def forward(self, x):
        vae_decoded, mu, log_var = self.vae(x)

        shape1 = vae_decoded.shape[-1]
        # vae_decoded = vae_decoded.reshape(-1, 1, shape1)

        vae_decoded = vae_decoded.reshape(-1, shape1)

        validity = self.discriminator(vae_decoded)

        # vae_decoded = vae_decoded.reshape(-1, shape1)

        return vae_decoded, mu, log_var, validity


# 定义损失函数
def loss_function(recon_x, x, mu, log_var):
    # BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # 如果数据不在 [0, 1] 之间，使用 MSE
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # return BCE + KLD
    return MSE + KLD


def gan_loss(validity, target):
    return nn.functional.cross_entropy(validity, target)


# 模型、优化器和损失函数
vae_gan = VAE_GAN(input_dim, hidden_dim)
dis_optimizer = optim.Adam(vae_gan.discriminator.parameters(), lr=lr)
vae_optimizer = optim.Adam(vae_gan.vae.parameters(), lr=lr)

data_final = torch.tensor(data_final, dtype=torch.float32)
# data_fu = torch.tensor(data_fu, dtype=torch.float32)
# data_fu_x = torch.tensor(data_fu_x, dtype=torch.float32)

# # 首先在训练开始时打乱数据
# data_fu_x = data_fu_x[torch.randperm(data_fu_x.shape[0])]

vae_gan.train()

# 保存列表，准备画图
discriminator_losses = []
vae_cos_losses = []

# --------------------------------------------------------------
batch_size = data_final.shape[0]
# --------------------------------------------------------------

# 初始化最小loss为无穷大
fid_min = float('inf')
dis_loss_max = float('-inf')
temp_loss_dis = 0.0
avg_distance = 0.0
average_cosine_similarity = 0.0
alpha = 1.0
patience = 10  # patience 的值，比如设置为 5 个 epoch
loss1_history = []  # 用于记录最近的 loss1
best_generated_data_numpy = None
# 训练循环
for epoch in range(epochs):
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)

    # # 1. 使用VAE生成伪造数据
    # VAE-GAN前向传播
    x_reconstructed, z_mean, z_logvar, validity = vae_gan(data_final)

    # 2. 生成样本
    z_real = data_final  # 真实小样本类数据的特征
    z_generated = x_reconstructed  # VAE生成的数据的特征

    # 判别器对真实数据的损失 (真实样本的标签为1)
    real_labels = torch.ones(batch_size, dtype=torch.long)
    gan_loss_value_real = gan_loss(vae_gan.discriminator(z_real), real_labels)

    # 判别器对生成数据的损失 (生成的假样本的标签为0)
    fake_labels = torch.zeros(batch_size, dtype=torch.long)
    gan_loss_value_fake = gan_loss(vae_gan.discriminator(z_generated), fake_labels)

    # 辨别器总损失
    loss_discriminator = gan_loss_value_fake + gan_loss_value_real

    # 生成器损失
    loss_generator = loss_function(z_generated, z_real, z_mean, z_logvar)
    loss_generator_gan = gan_loss(vae_gan.discriminator(z_generated), real_labels)

    loss_gen_total = loss_generator + loss_generator_gan

    # 反向传播和优化判别器
    dis_optimizer.zero_grad()
    loss_discriminator.backward(retain_graph=True)
    dis_optimizer.step()

    # 反向传播和优化生成器（与对比学习损失）
    vae_optimizer.zero_grad()
    loss_gen_total.backward()
    vae_optimizer.step()

    # torch.autograd.set_detect_anomaly(True)

    # 保存损失值
    discriminator_losses.append(loss_discriminator.item())
    vae_cos_losses.append(loss_gen_total.item() / 100)

    # 打印训练过程中的损失
    print(f"Epoch [{epoch + 1}/{epochs}], discriminator Loss: {loss_discriminator.item():.4f}")
    print(f"Epoch [{epoch + 1}/{epochs}], vae_cos Loss: {loss_gen_total.item() / 1000:.4f}")
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # 生成1000条新数据
    num_samples = 4000
    generated_data = generate_data(vae_gan, num_samples=num_samples, latent_dim=hidden_dim)

    # 转换为numpy数组并保存为csv文件
    generated_data_numpy = generated_data.numpy()

    # # np.savetxt('vae_gan_contra_generated_data.csv', generated_data_numpy, delimiter=',')
    # np.savetxt('vae_gan_generated_data.csv', generated_data_numpy, delimiter=',')
    # data_generate = pd.read_csv(
    #     'D:/python/pythonProject/PHDfirstTest/x_iiot/vae_gan_generated_data.csv', encoding='utf-8',
    #     low_memory=False, header=None, dtype='float64')
    # # 计算真实样本的均值和协方差
    mu_real, sigma_real = calculate_statistics(data_text)

    # 计算生成样本的均值和协方差
    mu_gen, sigma_gen = calculate_statistics(generated_data_numpy)

    # 计算 FID
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    if fid_value < fid_min:
        fid_min = fid_value
        best_generated_data_numpy = generated_data_numpy
        # # 计算生成样本之间的欧氏距离
        # distances = pdist(data_generate, metric='euclidean')
        # avg_distance = np.mean(distances)
        # # 计算样本之间的余弦相似度
        # cosine_sim_matrix = cosine_similarity(data_generate)
        # # 去除对角线元素（每个样本与自己比较的相似度为 1），只保留两两不同样本之间的相似度
        # cosine_sim_matrix_no_diag = cosine_sim_matrix - np.eye(cosine_sim_matrix.shape[0])
        # # 将矩阵合并为一个值：取所有相似度值的平均值
        # average_cosine_similarity = np.mean(cosine_sim_matrix_no_diag[cosine_sim_matrix_no_diag != 0])
        #
        # np.savetxt('best_vae_gan_generated_data.csv', generated_data_numpy, delimiter=',')
        #
        # # # 保存模型，保存模型的状态字典
        # # torch.save(vae_gan.state_dict(),
        # #            'D:/python/pythonProject/PHDfirstTest/UNSW/vae_gan_contra_lossMethod_generated_best_model.pth')

np.savetxt('D:/python/pythonProject/PHDfirstTest/x_iiot/VAE-GAN/6_4000_vae_gan_generated_data.csv',
           best_generated_data_numpy, delimiter=',')

import matplotlib.pyplot as plt

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.plot(vae_cos_losses, label='VAE Cos Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator Loss vs VAE Cos Loss')
plt.legend()
plt.show()

# # 生成新数据
# def generate_data(model, num_samples, latent_dim):
#     model.eval()
#     with torch.no_grad():
#         z = torch.randn(num_samples, latent_dim)
#         generated_data = model.vae.decoder(z)
#     return generated_data
#

print(f"最佳FID between real and generated samples：{fid_min}")
# print(f"最佳Average distance between generated samples：{avg_distance}")
# print(f"最佳生成样本之间的平均余弦相似度：{average_cosine_similarity}")

# # 生成1000条新数据
# num_samples = 21
# generated_data = generate_data(vae_gan, num_samples=num_samples, latent_dim=hidden_dim)
#
# # 转换为numpy数组并保存为csv文件
# generated_data_numpy = generated_data.numpy()
# import numpy as np
#
# # np.savetxt('vae_gan_contra_generated_data.csv', generated_data_numpy, delimiter=',')
# np.savetxt('vae_gan_generated_data.csv', generated_data_numpy, delimiter=',')
# # np.savetxt('vae_gan_contra_ContrastLoss_generated_data.csv', generated_data_numpy, delimiter=',')
#
# # np.savetxt('vae_gan_contra_lossMethod_generated_data.csv', generated_data_numpy, delimiter=',')
