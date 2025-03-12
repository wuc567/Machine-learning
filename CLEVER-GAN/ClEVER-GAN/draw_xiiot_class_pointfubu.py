# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
#
# # 读取每个类别的数据
# data_0 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/0_class_data.csv', encoding='utf-8', low_memory=False)
# data_1 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/1_class_data.csv', encoding='utf-8', low_memory=False)
# data_2 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/2_class_data.csv', encoding='utf-8', low_memory=False)
# data_3 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/3_class_data.csv', encoding='utf-8', low_memory=False)
# data_4 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/4_class_data.csv', encoding='utf-8', low_memory=False)
# data_5 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/5_class_data.csv', encoding='utf-8', low_memory=False)
# data_6 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/6_class_data.csv', encoding='utf-8', low_memory=False)
# data_7 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/7_class_data.csv', encoding='utf-8', low_memory=False)
# data_8 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/8_class_data.csv', encoding='utf-8', low_memory=False)
# data_9 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/little_class_data.csv', encoding='utf-8',
#                      low_memory=False)
#
#
# dataset_vae_9 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/VAE-GAN/9_5960_vae_gan_generated_data.csv',
#     encoding='utf-8',
#     low_memory=False)
# dataset_vae_8 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/VAE-GAN/8_1069_vae_gan_generated_data.csv',
#     encoding='utf-8',
#     low_memory=False)
# dataset_vae_7 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/VAE-GAN/7_2632_vae_gan_generated_data.csv',
#     encoding='utf-8',
#     low_memory=False)
# dataset_vae_6 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/VAE-GAN/6_842_vae_gan_generated_data.csv',
#     encoding='utf-8',
#     low_memory=False)
#
#
# dataset_gen_9 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/9_5960_best_vae_gan_contra_generated_data.csv',
#     encoding='utf-8',
#     low_memory=False)
#
# dataset_gen_8 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/8_1069_best_vae_gan_contra_generated_data.csv',
#     encoding='utf-8',
#     low_memory=False)
# dataset_gen_7 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/7_2632_best_vae_gan_contra_generated_data.csv',
#     encoding='utf-8',
#     low_memory=False)
# dataset_gen_6 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/6_842_best_vae_gan_contra_generated_data.csv',
#     encoding='utf-8',
#     low_memory=False)
#
# # 合并所有数据到一个 DataFrame，并手动添加标签
# data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, dataset_vae_9,
#              dataset_gen_9]
# class_centroids = []
#
#
#
#
#
# labels = []
#
# # 将所有数据合并为一个 DataFrame
# for i, data in enumerate(data_list):
# #     # 获取所有特征列
#      features = data.values  # 获取所有特征数据
#      labels.extend([i] * len(features))  # 为每个类别的数据添加标签
# #
# # 将所有特征数据合并成一个大矩阵
# all_features = np.vstack([data.values for data in data_list])
# #
# # 将标签转化为 numpy 数组
# labels = np.array(labels)
#
#
#
# # 使用 t-SNE 降维到 2D
# tsne = TSNE(n_components=2, random_state=42)
# reduced_features = tsne.fit_transform(all_features)
#
# # 可视化 10 个类别的分布情况
# plt.figure(figsize=(10, 8))
#
# # 遍历10个类别并使用不同颜色绘制每个类别的样本
# # for i in range(10):
# #     class_samples = reduced_features[labels == i]
# #     plt.scatter(class_samples[:, 0], class_samples[:, 1], label=f'Class {i}', alpha=0.6, s=50)
#
# for i in range(12):
#      class_samples = reduced_features[labels == i]
#      plt.scatter(class_samples[:, 0], class_samples[:, 1], label=f'Class {i}', alpha=0.6, s=2)
#
# # 添加图例和标题
# plt.title('Feature Distribution (t-SNE)')
# plt.legend()
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.grid(True)
#
# # 显示图形
# plt.show()
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 读取每个类别的数据
data_0 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/0_class_data.csv', encoding='utf-8', low_memory=False)
data_1 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/1_class_data.csv', encoding='utf-8', low_memory=False)
data_2 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/2_class_data.csv', encoding='utf-8', low_memory=False)
data_3 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/3_class_data.csv', encoding='utf-8', low_memory=False)
data_4 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/4_class_data.csv', encoding='utf-8', low_memory=False)
data_5 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/5_class_data.csv', encoding='utf-8', low_memory=False)
data_6 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/6_class_data.csv', encoding='utf-8', low_memory=False)
data_7 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/7_class_data.csv', encoding='utf-8', low_memory=False)
data_8 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/8_class_data.csv', encoding='utf-8', low_memory=False)
data_9 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/little_class_data.csv', encoding='utf-8',
                     low_memory=False)

dataset_vae_9 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/x_iiot/VAE-GAN/9_5960_vae_gan_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_vae_8 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/x_iiot/VAE-GAN/8_1069_vae_gan_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_vae_7 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/x_iiot/VAE-GAN/7_2632_vae_gan_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_vae_6 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/x_iiot/VAE-GAN/6_842_vae_gan_generated_data.csv',
    encoding='utf-8',
    low_memory=False)

dataset_gen_9 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/9_5960_best_vae_gan_contra_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_8 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/8_1069_best_vae_gan_contra_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_7 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/7_2632_best_vae_gan_contra_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_6 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_TRUE/6_842_best_vae_gan_contra_generated_data.csv',
    encoding='utf-8',
    low_memory=False)

# 合并所有数据到一个 DataFrame，并手动添加标签
# data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9,
#              dataset_gen_6, dataset_vae_6]
# data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9,
#              dataset_gen_7, dataset_vae_7]

data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, dataset_gen_9]
# data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9]

labels = []

# 将所有数据合并为一个 DataFrame
for i, data in enumerate(data_list):
    features = data.values  # 获取所有特征数据
    labels.extend([i] * len(features))  # 为每个类别的数据添加标签

# 将所有特征数据合并成一个大矩阵
all_features = np.vstack([data.values for data in data_list])

# 将标签转化为 numpy 数组
labels = np.array(labels)

# 使用 t-SNE 降维到 2D
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(all_features)

# 可视化 12 个类别的分布情况
plt.figure(figsize=(10, 8))

# 使用 tab20 调色板，确保每个类别颜色唯一
colors = plt.cm.tab20(np.linspace(0, 1, len(data_list))) * 0.8

print("color0", colors[0])
print("color1", colors[1])
print("color2", colors[2])
print("color3", colors[3])
print("color4", colors[4])
print("color5", colors[5])
print("color6", colors[6])
print("color7", colors[7])
print("color8", colors[8])
print("color9", colors[9])
# print("color10", colors[10])
# print("color11", colors[11])

# 修改 class11（第12个类别）的颜色为深紫色
# temp = colors[10]
# colors[11] = colors[2]
# colors[10] = (0.2, 0.1, 0.5, 1.0)  # 深紫色
# #
# colors[11] = [1.0, 0.73333333, 0.47058824, 1.0]
colors[2] = [0.09019608, 0.74509804, 0.81176471, 1.0]

# 遍历类别并绘制
for i in range(len(data_list)):
    class_samples = reduced_features[labels == i]
    plt.scatter(class_samples[:, 0], class_samples[:, 1],
                label=f'Class {i}',
                alpha=0.7,
                s=2,
                color=colors[i])

# 添加图例和标题
plt.title('Feature Distribution (t-SNE)')

legend = plt.legend()

# 调整图例中的颜色点大小
for handle in legend.legendHandles:
    handle.set_sizes([180])  # 设置图例点的大小（数值越大点越大）
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)

# 显示图形
plt.show()
