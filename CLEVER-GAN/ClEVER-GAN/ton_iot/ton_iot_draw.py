import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import pairwise_distances
import networkx as nx

# 读取每个类别的数据
data_0 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/0_class_data.csv', encoding='utf-8',
                     low_memory=False)
data_1 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/1_class_data.csv', encoding='utf-8',
                     low_memory=False)
data_2 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/2_class_data.csv', encoding='utf-8',
                     low_memory=False)
data_3 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/3_class_data.csv', encoding='utf-8',
                     low_memory=False)
data_4 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/4_class_data.csv', encoding='utf-8',
                     low_memory=False)
data_5 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/5_class_data.csv', encoding='utf-8',
                     low_memory=False)
data_6 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/6_class_data.csv', encoding='utf-8',
                     low_memory=False)
data_7 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/little_class_data.csv', encoding='utf-8',
                     low_memory=False)

# d6 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_0_6_test/6_4000_best_vae_gan_contra_generated_data.csv',
#     encoding='utf-8', low_memory=False)


dataset_vae_7 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/ton_iot/VAE-GAN/7_7783_vae_gan_generated_data.csv',
    encoding='utf-8',
    low_memory=False)

#
# dataset_gen_6 = pd.read_csv(
#     'D:/python/pythonProject/PHDfirstTest/x_iiot/CVG_0_6_test/6_842_best_vae_gan_contra_generated_data.csv',
#     encoding='utf-8',
#     low_memory=False)


dataset_gen_7 = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG_TRUE/7_7783_best_vae_gan_contra_generated_data.csv',
    encoding='utf-8',
    low_memory=False)

# 合并所有数据到一个 DataFrame，并手动添加标签
# data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9]
# data_list = [data_0, dataset_gen_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9]
# data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, dataset_gen_9]
# data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, dataset_gen_7]
data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7]
class_centroids = []

labels = []

# 将所有数据合并为一个 DataFrame
for i, data in enumerate(data_list):
    #     # 获取所有特征列
    features = data.values  # 获取所有特征数据
    labels.extend([i] * len(features))  # 为每个类别的数据添加标签
#
# 将所有特征数据合并成一个大矩阵
all_features = np.vstack([data.values for data in data_list])
#
# 将标签转化为 numpy 数组
labels = np.array(labels)

#
# # 使用 t-SNE 降维到 3D
# tsne = TSNE(n_components=3, random_state=42)
# reduced_features = tsne.fit_transform(all_features)
#
# # 将结果转换为 DataFrame 以便于使用 Plotly
# reduced_df = pd.DataFrame(reduced_features, columns=['t-SNE 1', 't-SNE 2', 't-SNE 3'])
# reduced_df['Label'] = labels
#
# # 使用 Plotly 创建 3D 散点图
# fig = px.scatter_3d(reduced_df, x='t-SNE 1', y='t-SNE 2', z='t-SNE 3', color='Label',
#                     title='Feature Distribution of 11 Classes (t-SNE - 3D)',
#                     labels={'Label': 'Class'}, opacity=0.7)
#
# # 显示图形
# fig.show()

# 使用 t-SNE 降维到 2D
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(all_features)

# 可视化 10 个类别的分布情况
plt.figure(figsize=(10, 8))

# 遍历10个类别并使用不同颜色绘制每个类别的样本
# for i in range(10):
#     class_samples = reduced_features[labels == i]
#     plt.scatter(class_samples[:, 0], class_samples[:, 1], label=f'Class {i}', alpha=0.6, s=50)

for i in range(8):
    class_samples = reduced_features[labels == i]
    plt.scatter(class_samples[:, 0], class_samples[:, 1], label=f'Class {i}', alpha=0.6, s=2)

# 添加图例和标题
plt.title('Feature Distribution(t-SNE)')
legend = plt.legend()

# 调整图例中的颜色点大小
for handle in legend.legendHandles:
    handle.set_sizes([180])  # 设置图例点的大小（数值越大点越大）

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)

# 显示图形
plt.show()

# # 使用 t-SNE 降维到 3D
# tsne = TSNE(n_components=2, random_state=42)
# reduced_features = tsne.fit_transform(all_features)
#
# # 假设 reduced_features 是你的二维降维结果，labels 是类别标签
# # 计算每个类别的均值特征向量
# unique_labels = np.unique(labels)
# class_centroids = []
#
# for label in unique_labels:
#     class_centroids.append(np.mean(reduced_features[labels == label], axis=0))
#
# # 计算类别间的成对欧式距离
# distance_matrix = pairwise_distances(class_centroids, metric='euclidean')
#
# # 绘制热力图
# plt.figure(figsize=(8, 6))
# sns.heatmap(distance_matrix, annot=True, cmap='coolwarm', xticklabels=unique_labels, yticklabels=unique_labels)
# plt.title('Pairwise Distance Between Classes')
# plt.xlabel('Class')
# plt.ylabel('Class')
# plt.show()

# A = data_0
# B = d6
#
# A = A.to_numpy()
# B = B.to_numpy()
#
# # 计算类别A和类别B的均值向量
# mean_A = np.mean(A, axis=0)
# mean_B = np.mean(B, axis=0)
#
# # 计算这两个均值向量之间的余弦相似度
# cos_sim = cosine_similarity([mean_A], [mean_B])
#
# print(f"Cosine similarity between Category A and Category B: {cos_sim[0][0]}")
