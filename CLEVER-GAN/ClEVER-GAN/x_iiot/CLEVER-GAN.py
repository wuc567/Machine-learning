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
from scipy import linalg
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity

pos_sim_list = []
neg_sim_list = []


def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也要设置随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内置随机数生成器的随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作的结果确定
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化算法选择


# set_seed(42)  # 设置为固定的随机种子，例如42
set_seed(1234)  # 设置为固定的随机种子，例如42

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 生成新数据
def generate_data(model, num_samples, latent_dim):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_data = model.decoder(z)
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

print(data_text.shape)
print(data_text)

data = df.to_numpy()
feature = data[:, 0:-3]
lable = data[:, -2]

print(lable.shape)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 你要根据第二列的值来筛选行
value_to_find = 6
# ----------------------------------------------------------------------------------------------------------------------------------------------

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
# ----------------------------------------------------------------------------------------------------------------------------------------------
value_to_remove = 6
# ----------------------------------------------------------------------------------------------------------------------------------------------

# 使用布尔索引来筛选出不包含指定值的行
filtered_data = data_fu[data_fu[:, column_index] != value_to_remove]

# 计算倒数第一列和倒数第三列的索引
index_last = -1  # 倒数第一列
index_third_last = -3  # 倒数第三列

# 使用 numpy.delete() 删除倒数第一列和倒数第三列
data_fu = np.delete(filtered_data, [index_third_last, index_last], axis=1)

print("data_fu", data_fu.shape)

# data_fu_x = data_fu[:39963, :-1]  # 标签为9的选择
# data_fu_y = data_fu[:39963, -1]  # 标签为9的选择

# data_fu_x = data_fu[:31801, :-1]  # 标签为0的选择
# data_fu_y = data_fu[:31801, -1]  # 标签为0的选择
#
# data_fu_x = data_fu[:37672, :-1]  # 标签为1的选择
# data_fu_y = data_fu[:37672, -1]  # 标签为1的选择

# data_fu_x = data_fu[:29060, :-1]  # 标签为4的选择
# data_fu_y = data_fu[:29060, -1]  # 标签为4的选择

# data_fu_x = data_fu[:34713, :-1]  # 标签为5的选择
# data_fu_y = data_fu[:34713, -1]  # 标签为5的选择

data_fu_x = data_fu[:28585, :-1]  # 标签为6的选择
data_fu_y = data_fu[:28585, -1]  # 标签为6的选择

# data_fu_x = data_fu[:33633, :-1]  # 标签为7的选择
# data_fu_y = data_fu[:33633, -1]  # 标签为7的选择

# data_fu_x = data_fu[:32496, :-1]  # 标签为8的选择
# data_fu_y = data_fu[:32496, -1]  # 标签为8的选择

# data_fu_x = data_fu[:39861, :-1]  # 标签为3的选择
# data_fu_y = data_fu[:39861, -1]  # 标签为3的选择

print(data_fu_x)
print(data_fu_x.shape)
print(data_fu_y)
print(data_fu_y.shape)

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
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
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
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(57, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out_x = self.fc3(x.clone())

        return out_x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_real, z_generated, z_negative):
        # 正样本对的相似度
        pos_sim = torch.exp(F.cosine_similarity(z_real, z_generated) / self.temperature)
        print("正样本相似度：", pos_sim.mean())
        pos_sim_list.append(pos_sim.mean().item())

        # 负样本对的相似度（z_negative 为其他类别数据）
        neg_sim = torch.exp(
            F.cosine_similarity(z_generated.unsqueeze(1), z_negative.unsqueeze(0), dim=-1) / self.temperature)
        print("负样本相似度：", neg_sim.mean())
        neg_sim_list.append(neg_sim.mean().item())

        # 对比损失
        loss = -torch.log(pos_sim / (pos_sim + neg_sim.sum(dim=-1)))
        # loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        return loss.mean(), neg_sim.mean()


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
# vae_gan = VAE_GAN(input_dim, hidden_dim)
print("666")
vae = VAE(input_dim, hidden_dim).to(device)
discriminator = Discriminator().to(device)
print("666")

contrastive_loss = ContrastiveLoss()
dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
vae_optimizer = optim.Adam(vae.parameters(), lr=0.001)

data_final = (torch.tensor(data_final, dtype=torch.float32)).to(device)
data_fu = (torch.tensor(data_fu, dtype=torch.float32)).to(device)
data_fu_x = (torch.tensor(data_fu_x, dtype=torch.float32)).to(device)

# 首先在训练开始时打乱数据
data_fu_x = data_fu_x[torch.randperm(data_fu_x.shape[0])]

vae.train()
discriminator.train()

# 保存列表，准备画图
discriminator_losses = []
vae_cos_losses = []

# 初始化最小loss为无穷大
fid_min = float('inf')
dis_loss_max = float('-inf')
temp_loss_dis = 0.0
avg_distance = 0.0
average_cosine_similarity = 0.0
alpha = 1.0
patience = 10  # patience 的值，比如设置为 10 个 epoch
coscos = []

loss1_history = []  # 用于记录最近的 loss1
best_generated_data_numpy = None
# 训练循环
for epoch in range(epochs):
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)

    # # 1. 使用VAE生成伪造数据
    # VAE-GAN前向传播

    x_reconstructed, z_mean, z_logvar = vae(data_final)

    # 2. 构建正样本对（真实数据与VAE生成数据配对）
    z_real = data_final  # 真实小样本类数据的特征
    z_generated = x_reconstructed  # VAE生成的数据的特征

    # print("正样本：", z_generated)

    # # 3. 随机采样负样本对（从其他类别数据中抽取一部分）
    # batch_size = data_final.size(0)
    # negative_indices = torch.randint(0, data_fu_x.shape[0], (batch_size,))
    # z_negative = data_fu_x[negative_indices]  # 负样本来自其他类别的数据

    # 3. 顺序采样负样本对
    batch_size = data_final.size(0)
    start_idx = (epoch * batch_size) % data_fu_x.shape[0]  # 计算每个epoch的起始位置
    end_idx = start_idx + batch_size  # 取21条
    z_negative = data_fu_x[start_idx:end_idx]  # 顺序取样

    # print("负样本：",z_negative)

    # 4. 计算对比损失（这里是对比损失）
    cos_loss, neg_loss = contrastive_loss(z_real, z_generated, z_negative)

    # 判别器对真实数据的损失 (真实样本的标签为1)
    real_labels = torch.ones(batch_size, dtype=torch.long).to(device)

    # z_real_cnn = z_real.reshape(-1, 1, 57)
    # z_generated_cnn = z_generated.reshape(-1, 1, 57)

    # gan_loss_value_real = gan_loss(vae_gan.discriminator(z_real_cnn), real_labels)
    dis1 = discriminator(z_real)
    gan_loss_value_real = gan_loss(dis1, real_labels)

    # 判别器对生成数据的损失 (生成的假样本的标签为0)
    fake_labels = torch.zeros(batch_size, dtype=torch.long).to(device)

    # gan_loss_value_fake = gan_loss(vae_gan.discriminator(z_generated_cnn), fake_labels)

    dis2 = discriminator(z_generated)
    gan_loss_value_fake = gan_loss(dis2, fake_labels)

    # 辨别器总损失
    loss_discriminator = gan_loss_value_fake + gan_loss_value_real

    # 生成器损失

    loss_generator_vae = loss_function(z_generated, z_real, z_mean, z_logvar)
    loss_generator_gan = gan_loss(dis2, real_labels)

    # 反向传播和优化判别器
    dis_optimizer.zero_grad()
    loss_discriminator.backward(retain_graph=True)
    dis_optimizer.step()

    # 生成器vae损失和对比学习损失
    # 赋予权重，目的是为了让vae生成器的loss主导
    # wight1 = loss_generator / (loss_generator + cos_loss)
    # wight2 = cos_loss / (loss_generator + cos_loss)
    # if wight1 > wight2:
    #     loss_cos_vae = wight1 * loss_generator + wight2 * cos_loss
    # else:
    #     loss_cos_vae = wight2 * loss_generator + wight1 * cos_loss

    # loss_cos_vae = loss_generator_vae + loss_generator_gan
    # loss_cos_vae = loss_generator_vae + loss_generator_gan + alpha * cos_loss

    # print("loss_gen_vae:", loss_generator_vae)
    # print("loss_gen_gan:", loss_generator_gan)
    # print("cos_loss:", cos_loss)
    loss_cos_vae = 0.00001 * loss_generator_vae + loss_generator_gan + 0.1 * alpha * cos_loss
    # loss_cos_vae = 0.00001 * loss_generator_vae + loss_generator_gan + 0.1 * cos_loss
    # loss_cos_vae = 0.00001 * loss_generator_vae + loss_generator_gan

    # 保存当前 epoch 的 loss1
    # loss1_history.append(loss_generator_vae)
    # loss1_history.append(loss_generator_gan)
    loss1_history.append(loss_generator_vae)

    # 保持 loss1_history 的长度为 patience
    if len(loss1_history) > patience:
        loss1_history.pop(0)  # 移除最早的 loss1

    # 检查 patience 个 epoch 内的 loss1 是否有下降
    if len(loss1_history) == patience and all(x >= loss1_history[0] for x in loss1_history[1:]):
        alpha = alpha * 0.95  # 如果 loss1 没有下降，alpha 减少
        # if alpha < 0.2:
        #     alpha = 0.2
        print(f"Loss1 has not decreased for {patience} epochs, reducing alpha to {alpha:.4f}")

    # 反向传播和优化生成器（与对比学习损失）
    vae_optimizer.zero_grad()
    loss_cos_vae.backward()
    vae_optimizer.step()

    # torch.autograd.set_detect_anomaly(True)

    # 保存损失值
    discriminator_losses.append(loss_discriminator.item())
    vae_cos_losses.append(loss_cos_vae.item())
    coscos.append(cos_loss.item())

    # 打印训练过程中的损失
    print(f"Epoch [{epoch + 1}/{epochs}], discriminator Loss: {loss_discriminator.item():.4f}")
    print(f"Epoch [{epoch + 1}/{epochs}], vae_cos Loss: {loss_cos_vae.item():.4f}")

    # # 生成3000条新数据
    # num_samples = 2632
    # generated_data = generate_data(vae, num_samples=num_samples, latent_dim=hidden_dim)
    # # 转换为numpy数组并保存为csv文件
    # generated_data_numpy = generated_data.cpu().numpy()
    # # np.savetxt('0_10000_vae_gan_contra_lossMethod_generated_data.csv', generated_data_numpy, delimiter=',')
    # #
    # # # 转换为numpy数组并保存为csv文件
    # # generated_data_numpy = generated_data.cpu().numpy()
    #
    # # # np.savetxt('vae_gan_contra_generated_data.csv', generated_data_numpy, delimiter=',')
    # # np.savetxt('9_10000_vae_gan_contra_lossMethod_generated_data.csv', generated_data_numpy, delimiter=',')
    # # # data_generate = pd.read_csv(
    # # #     'D:/python/pythonProject/PHDfirstTest/x_iiot/vae_gan_contra_generated_data.csv', encoding='utf-8',
    # # #     low_memory=False, header=None, dtype='float64')
    # # data_generate = pd.read_csv(
    # #     'D:/python/pythonProject/PHDfirstTest/x_iiot/9_10000_vae_gan_contra_lossMethod_generated_data.csv',
    # #     encoding='utf-8',
    # #     low_memory=False, header=None, dtype='float64')
    # # 计算真实样本的均值和协方差
    # mu_real, sigma_real = calculate_statistics(data_text)
    # #
    # # 计算生成样本的均值和协方差
    # mu_gen, sigma_gen = calculate_statistics(generated_data_numpy)
    #
    # # 计算 FID
    # fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    # if fid_value < fid_min:
    #     fid_min = fid_value
    #     print("fid_min", fid_min)
    #     best_generated_data_numpy = generated_data_numpy
    #     # # 计算生成样本之间的欧氏距离
    #     # distances = pdist(generated_data_numpy, metric='euclidean')
    #     # avg_distance = np.mean(distances)
    #     # # 计算样本之间的余弦相似度
    #     # cosine_sim_matrix = cosine_similarity(generated_data_numpy)
    #     # # 去除对角线元素（每个样本与自己比较的相似度为 1），只保留两两不同样本之间的相似度
    #     # cosine_sim_matrix_no_diag = cosine_sim_matrix - np.eye(cosine_sim_matrix.shape[0])
    #     # # 将矩阵合并为一个值：取所有相似度值的平均值
    #     # average_cosine_similarity = np.mean(cosine_sim_matrix_no_diag[cosine_sim_matrix_no_diag != 0])
    #
    #     # # 保存模型，保存模型的状态字典
    #     # torch.save(vae_gan.state_dict(),
    #     #            'D:/python/pythonProject/PHDfirstTest/UNSW/vae_gan_contra_lossMethod_generated_best_model.pth')
    #
    # # # 记录并检查辨别器的损失值
    # # if loss_discriminator.item() > dis_loss_max:
    # #     dis_loss_max = loss_discriminator.item()
    # #     temp_loss_dis = dis_loss_max
    # #     # 生成3000条新数据
    # #     num_samples = 5960
    # #     generated_data = generate_data(vae, num_samples=num_samples, latent_dim=hidden_dim)
    # #     # 转换为numpy数组并保存为csv文件
    # #     generated_data_numpy = generated_data.cpu().numpy()

# np.savetxt(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/7_2632_best_vae_gan_contra_generated_data.csv',
#     best_generated_data_numpy, delimiter=',')


# np.savetxt(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/7_2632_vae_gan_contra_generated_data.csv',
#     best_generated_data_numpy, delimiter=',')


# print("最大disloss：", temp_loss_dis)
# 生成3000条新数据
# num_samples = 5960
# generated_data = generate_data(vae, num_samples=num_samples, latent_dim=hidden_dim)
# # 转换为numpy数组并保存为csv文件
# generated_data_numpy = generated_data.cpu().numpy()
# np.savetxt(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CLVAE-GAN-noStrategy/6_842_best_vae_gan_contra_lossMethod_generated_data.csv',
#     generated_data_numpy, delimiter=',')

# np.savetxt(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CLVAE-GAN/9_5960_best_vae_gan_contra_lossMethod_generated_data.csv',
#     generated_data_numpy, delimiter=',')

import matplotlib.pyplot as plt

# # 绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.plot(discriminator_losses, label='Discriminator Loss')
# plt.plot(vae_cos_losses, label='VAE(no ECLG) Loss ')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Discriminator Loss vs VAE(no ECLG) Loss')
# plt.legend()
# plt.show()


# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.plot(vae_cos_losses, label='VAE Loss ')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator Loss vs VAE Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(coscos, label='Contrastive Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Contrastive Loss')
plt.legend()
plt.show()

print(f"最佳FID between real and generated samples：{fid_min}")
# print(f"最佳Average distance between generated samples：{avg_distance}")
# print(f"最佳生成样本之间的平均余弦相似度：{average_cosine_similarity}")


epochs = list(range(1, 5001))  # 假设5000个epoch
# 创建一个包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# 绘制正样本相似度随epoch变化的曲线
ax1.plot(epochs, pos_sim_list, label='Positive Similarity', color='blue', linewidth=2)
ax1.set_title('Positive Similarity vs. Epoch', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Similarity', fontsize=14)
ax1.grid(True)
ax1.legend()

# 绘制负样本相似度随epoch变化的曲线
ax2.plot(epochs, neg_sim_list, label='Negative Similarity', color='red', linewidth=2)
ax2.set_title('Negative Similarity vs. Epoch', fontsize=16)
ax2.set_xlabel('Epoch', fontsize=14)
ax2.set_ylabel('Similarity', fontsize=14)
ax2.grid(True)
ax2.legend()

# 显示图形
plt.tight_layout()  # 自动调整子图间距
plt.show()

# 将 pos_sim_list 和 neg_sim_list 组合成 DataFrame
df = pd.DataFrame({
    'Positive Similarity': pos_sim_list,
    'Negative Similarity': neg_sim_list
})

# # 将 DataFrame 保存为 .txt 文件
# df.to_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/List_txt_True/Xiiot_6_no_strategy_similarity_lists.txt',
#           sep='\t', index=False)

# # 将 DataFrame 保存为 .txt 文件
# df.to_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/List_txt_True/Xiiot_9_similarity_lists.txt',
#           sep='\t', index=False)
