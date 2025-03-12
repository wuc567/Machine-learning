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
        gen_labels = torch.randint(0, class_num, (num_samples,)).to(device)
        generated_data = model(z, gen_labels)
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


# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/little_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/6_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/5_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/4_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/2_class_data.csv', encoding='utf-8',
#                         low_memory=False)
# data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/1_class_data.csv', encoding='utf-8',
#                         low_memory=False)
data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/0_class_data.csv', encoding='utf-8',
                        low_memory=False)
df = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/datasets_ton_iot_afterProcess.csv', encoding='utf-8',
                 low_memory=False)

data = df.to_numpy()

feature = data[:, 0:-1]
lable = data[:, -1]

print(feature.shape)
print(lable.shape)

# ---------------------------------------------------------------------------------------------------------
# 你要根据第二列的值来筛选行
value_to_find = 0
# ---------------------------------------------------------------------------------------------------------


# 使用布尔索引筛选行
filtered_rows = data[data[:, -1] == value_to_find]

real_data = filtered_rows[:, 0:-1]
real_label = filtered_rows[:, -1]
real_label = real_label.reshape(-1, 1)
real_data = np.hstack((real_data, real_label))
# np.savetxt('little_class_data.csv', real_data, delimiter=',')


# real_label就是真实的小样本数据
print(real_data)
print(real_data.shape)


class GenderModel(nn.Module):
    def __init__(self, class_dim):
        super(GenderModel, self).__init__()
        self.label_emb = nn.Embedding(class_dim, class_dim)
        self.model = nn.Sequential(
            nn.Linear(100 + class_dim, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 500),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(500),
            nn.Linear(500, 106),

        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        input_data = torch.cat([noise, label_embedding], dim=1)
        return self.model(input_data)


class Discriminator(nn.Module):
    def __init__(self, class_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(106, 70),
            nn.ReLU(),
            nn.Linear(70, 106),
            nn.ReLU(),
            nn.Linear(106, 100),

        )
        # 输出真假性
        self.adv_layer = nn.Sequential(nn.Linear(100, 1), nn.Sigmoid())
        # 输出类别
        self.aux_layer = nn.Sequential(nn.Linear(100, class_dim), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.model(x)
        validity = self.adv_layer(x)  # 判别真假
        label = self.aux_layer(x)  # 分类
        return validity, label


# def gan_loss(validity, target):
#     return nn.functional.cross_entropy(validity, target)


# 损失函数
adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

# def gene_loss(data, target):
#     return nn.functional.mse_loss(data, target)

class_num = 8
generator = GenderModel(class_num).to(device)
discriminator = Discriminator(class_num).to(device)

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
    data_labels = real_data[:, -1].long()
    # for i in range(0, len(real_data), batch_size):
    # 1.先生成标签数据
    valid = torch.ones(batch_size, dtype=torch.long).to(device)
    fake = torch.zeros(batch_size, dtype=torch.long).to(device)

    # 2.训练辨别器
    # 随机生成样本
    # noise_data = torch.randn(batch_size, 100, 1)

    # 判别真实数据
    real_validity, dis_real_label = discriminator(real_data)
    valid = valid.view(-1, 1).float()
    fake = fake.view(-1, 1).float()
    d_real_loss = adversarial_loss(real_validity, valid) + auxiliary_loss(dis_real_label, data_labels)

    # 判别虚假数据
    noise_data = torch.randn(batch_size, 100).to(device)
    # gene_data = generator(noise_data)
    gen_labels = torch.randint(0, class_num, (batch_size,)).to(device)
    fake_data = generator(noise_data, gen_labels)

    fake_validity, fake_label = discriminator(fake_data.detach())
    d_fake_loss = adversarial_loss(fake_validity, fake) + auxiliary_loss(fake_label, gen_labels)

    # 辨别器总损失
    d_loss = (d_real_loss + d_fake_loss) / 2
    optimizer_d.zero_grad()
    d_loss.backward()
    optimizer_d.step()

    # -----------------
    #  训练生成器
    # -----------------

    fake_validity, pred_label = discriminator(fake_data)
    g_loss = adversarial_loss(fake_validity, valid) + auxiliary_loss(pred_label, gen_labels)

    optimizer_g.zero_grad()
    g_loss.backward()
    optimizer_g.step()

    # 打印训练过程中的损失
    print(f"Epoch [{epoch + 1}/{epochs}], discriminator Loss: {d_loss.item():.4f}")
    print(f"Epoch [{epoch + 1}/{epochs}], generator Loss: {g_loss.item():.4f}")
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------------

# 生成1000条新数据
num_samples = 7397
generated_data = generate_data(generator, num_samples=num_samples, latent_dim=100)

# 转换为numpy数组并保存为csv文件
generated_data_numpy = generated_data.cpu().numpy()
np.savetxt('D:/python/pythonProject/PHDfirstTest/ton_iot/ACGAN/0_7397_acgan_generated_data.csv', generated_data_numpy,
           delimiter=',')
