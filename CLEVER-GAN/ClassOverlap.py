import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 设置随机种子
np.random.seed(42)

# 生成两个类的重叠数据
class_1 = np.random.normal(loc=0, scale=1, size=(100, 2))
class_2 = np.random.normal(loc=1, scale=1, size=(100, 2))

# 将类重叠数据合并
data = np.vstack((class_1, class_2))
labels = np.array([0]*100 + [1]*100)

# 绘制特征空间图
plt.figure(figsize=(10, 6))

# 创建自定义色图
cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])

# 绘制散点图
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, edgecolor='k', s=50)

# 添加图例
plt.legend(handles=scatter.legend_elements()[0], labels=['Class 1', 'Class 2'], loc='upper right')

# 设置标题和轴标签
plt.title('Feature Space with Class Overlap')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 显示网格
plt.grid()

# 显示图形
plt.show()
