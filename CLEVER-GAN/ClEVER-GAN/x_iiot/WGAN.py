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

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 生成新数据
def generate_data(model, num_samples, latent_dim):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        # z = torch.randn(num_samples, latent_dim, 1)
        generated_data = model(z)
    return generated_data.reshape(num_samples, -1)


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


data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/0_class_data.csv', encoding='utf-8',
                        low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/1_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/4_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/5_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/6_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/7_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/8_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/3_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/little_class_data.csv', encoding='utf-8',
#                         low_memory=False)
df = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/datasets_xiiot_afterProcess.csv', encoding='utf-8',
                 low_memory=False)

data = df.to_numpy()

feature = data[:, 0:-3]
lable = data[:, -2]

print(lable.shape)

# 你要根据第二列的值来筛选行
value_to_find = 0

# 使用布尔索引筛选行
filtered_rows = data[data[:, -2] == value_to_find]

real_data = filtered_rows[:, 0:-3]

# data_final就是真实的小样本数据
print(real_data)
print(real_data.shape)


class GenderModel(nn.Module):
    def __init__(self):
        super(GenderModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 500),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(500),
            nn.Linear(500, 57),

        )

    def forward(self, x):
        x = self.model(x)
        # print("从反转卷积出来后：", x.shape)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(57, 70),
            nn.ReLU(),
            nn.Linear(70, 57),
            nn.ReLU(),
            nn.Linear(57, 1),

        )

    def forward(self, x):
        x = self.model(x)

        return x


def gan_loss(validity, target):
    return nn.functional.cross_entropy(validity, target)


# def gene_loss(data, target):
#     return nn.functional.mse_loss(data, target)


generator = GenderModel().to(device)
discriminator = Discriminator().to(device)

# 定义优化器
optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

epochs = 5000

generator.train()
discriminator.train()

real_data = torch.tensor(real_data, dtype=torch.float32).to(device)
print(real_data)
print(real_data.shape)



# 初始化最小loss为无穷大
fid_min = float('inf')
avg_distance = 0.0
average_cosine_similarity = 0.0
best_generated_data_numpy = None

# 训练循环
for epoch in range(epochs):
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)
    batch_size = real_data.shape[0]
    # for i in range(0, len(real_data), batch_size):
    # # 1.先生成标签数据
    # real_label = torch.ones(batch_size, dtype=torch.long).to(device)
    # fake_label = torch.zeros(batch_size, dtype=torch.long).to(device)

    # 2.训练辨别器
    # 随机生成样本
    # noise_data = torch.randn(batch_size, 100, 1)
    noise_data = torch.randn(batch_size, 100).to(device)
    gene_data = generator(noise_data)

    real_loss = discriminator(real_data).mean()
    fake_loss = discriminator(gene_data.detach()).mean()
    c_loss = -(real_loss - fake_loss)  # Wasserstein 距离

    # real_data_split = real_data[i:i + batch_size]



    optimizer_d.zero_grad()
    c_loss.backward(retain_graph=True)
    optimizer_d.step()

    # 2.训练生成器
    # 计算生成样本损失

    # print("gene_data",gene_data)

    noise_data = torch.randn(batch_size, 100).to(device)
    gene_data = generator(noise_data)
    g_loss = -discriminator(gene_data).mean()

    optimizer_g.zero_grad()
    g_loss.backward()
    optimizer_g.step()

    # 打印训练过程中的损失
    print(f"Epoch [{epoch + 1}/{epochs}], discriminator Loss: {c_loss.item():.4f}")
    print(f"Epoch [{epoch + 1}/{epochs}], generator Loss: {g_loss.item():.4f}")
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # 生成1000条新数据
    num_samples = 1939
    generated_data = generate_data(generator, num_samples=num_samples, latent_dim=100)

    # 转换为numpy数组并保存为csv文件
    generated_data_numpy = generated_data.cpu().numpy()

    # np.savetxt('vae_gan_contra_generated_data.csv', generated_data_numpy, delimiter=',')
    # np.savetxt('gan_generated_data.csv', generated_data_numpy, delimiter=',')
    # data_generate = pd.read_csv(
    #     'D:/python/pythonProject/PHDfirstTest/x_iiot/gan_generated_data.csv', encoding='utf-8',
    #     low_memory=False, header=None, dtype='float64')
    # 计算真实样本的均值和协方差
    mu_real, sigma_real = calculate_statistics(data_text)

    # 计算生成样本的均值和协方差
    mu_gen, sigma_gen = calculate_statistics(generated_data_numpy)

    # 计算 FID
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    if fid_value < fid_min:
        fid_min = fid_value
        best_generated_data_numpy = generated_data_numpy

np.savetxt('D:/python/pythonProject/PHDfirstTest/x_iiot/WGAN/0_1939_wgan_generated_data.csv', best_generated_data_numpy,
           delimiter=',')

print(f"最佳FID between real and generated samples：{fid_min}")
