import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 读取多个 CSV 文件并合并数据
file_paths = [
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG/7_7783_best_vae_gan_contra_lossMethod_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG/6_7439_best_vae_gan_contra_lossMethod_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG/5_4912_best_vae_gan_contra_lossMethod_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG/4_6804_best_vae_gan_contra_lossMethod_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG/2_7308_best_vae_gan_contra_lossMethod_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG/1_4149_best_vae_gan_contra_lossMethod_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG/0_7397_best_vae_gan_contra_lossMethod_generated_data.csv',

    'D:/python/pythonProject/PHDfirstTest/ton_iot/GAN/7_7783_best_gan_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG_withnoStrategy/7_7783_best_vae_gan_contra_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG_withnoStrategy/6_7439_best_vae_gan_contra_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG_withnoStrategy/5_4912_best_vae_gan_contra_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG_withnoStrategy/4_6804_best_vae_gan_contra_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG_withnoStrategy/2_7308_best_vae_gan_contra_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG_withnoStrategy/1_4149_best_vae_gan_contra_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/CVG_withnoStrategy/0_7397_best_vae_gan_contra_generated_data.csv',
    'D:/python/pythonProject/PHDfirstTest/ton_iot/0_class_data.csv',
    'D:/python/pythonProject/PHDfirstTest/ton_iot/1_class_data.csv',
    'D:/python/pythonProject/PHDfirstTest/ton_iot/2_class_data.csv',
    'D:/python/pythonProject/PHDfirstTest/ton_iot/4_class_data.csv',
    'D:/python/pythonProject/PHDfirstTest/ton_iot/5_class_data.csv',
    'D:/python/pythonProject/PHDfirstTest/ton_iot/6_class_data.csv',





    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/GAN/6_7439_best_gan_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/GAN/5_4912_best_gan_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/GAN/4_6804_best_gan_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/GAN/2_7308_best_gan_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/GAN/1_4149_best_gan_generated_data.csv',
    # 'D:/python/pythonProject/PHDfirstTest/ton_iot/GAN/0_7397_best_gan_generated_data.csv',

]

# 为每个类别分配颜色和标签
# colors = ['blue', 'green', 'orange', 'red', 'yellow']
colors = ['black', 'y', 'c', 'm', 'grey', 'pink', 'g']
# labels = ['Category 4', 'Category 3', 'Category 2', 'Category 1', 'Category 0']
labels = ['Category 7', 'Category 6', 'Category 5', 'Category 4', 'Category 3', 'Category 2', 'Category 1']

plt.figure(figsize=(10, 7))

# 对每个文件执行以下操作
for i, file_path in enumerate(file_paths):
    color = colors[i]
    label = labels[i]

    # 读取数据并检查
    data = pd.read_csv(file_path, encoding='utf-8', low_memory=False).dropna()
    data = data.select_dtypes(include=[np.number])  # 选择数值列
    features = data.values

    # 检查数据是否为空
    if features.size == 0:
        print(f"Warning: No data available in file {file_path}. Skipping.")
        continue

    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # 绘制该类别的特征点
    plt.scatter(features_2d[:, 0], features_2d[:, 1], s=10, color=color, alpha=0.5, label=label)

# 添加图例和标题
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('2D Visualization of Generated Data by Category (PCA)')
plt.legend()
plt.show()
