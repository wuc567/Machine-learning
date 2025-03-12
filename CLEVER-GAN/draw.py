# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 设置随机种子
# np.random.seed(0)
#
# # 网格点
# x = np.linspace(-3, 3, 100)
# y = np.linspace(-3, 3, 100)
# x, y = np.meshgrid(x, y)
#
# # 三维高斯分布的均值和协方差
# mean = [0, 0]
# cov = [[1, 0.5], [0.5, 1]]
#
# # 计算高斯分布
# pos = np.dstack((x, y))
# z = (1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))) * np.exp(-0.5 * np.einsum('...k,kl,...l->...', pos - mean, np.linalg.inv(cov), pos - mean))
#
# # 绘制三维表面图
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
#
# # 隐藏坐标轴
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
#
# # 保存图像
# plt.savefig("3d_normal_distribution_surface.png", bbox_inches='tight', transparent=True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子
np.random.seed(0)

# 网格点
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
x, y = np.meshgrid(x, y)

# 三维高斯分布的均值和协方差
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]

# 计算高斯分布
pos = np.dstack((x, y))
z = (1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))) * np.exp(-0.5 * np.einsum('...k,kl,...l->...', pos - mean, np.linalg.inv(cov), pos - mean))

# 绘制三维表面图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='plasma', alpha=0.8)  # 改为 'plasma' 颜色映射

# 隐藏坐标轴
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 保存图像
plt.savefig("3d_normal_distribution_surface.png", bbox_inches='tight', transparent=True)
plt.show()





