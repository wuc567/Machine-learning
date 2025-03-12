import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# 生成不平衡数据
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    class_sep=0.5,
    random_state=42,
)

# 训练 SVM 模型
clf = SVC(kernel="linear", probability=True, random_state=42)
clf.fit(X, y)

# 创建网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# 计算决策函数值
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(10, 8), dpi=300)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 7), cmap="coolwarm", alpha=0.3)
plt.contour(xx, yy, Z, levels=[0], colors="black", linestyles="--", linewidths=1.5)

# 绘制数据点
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="blue", alpha=0.8, label="Majority Class")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="red", alpha=0.8, label="Minority Class")

# 添加图例和标题
plt.title("Imbalanced Data with Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()
