import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也要设置随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内置随机数生成器的随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作的结果确定
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化算法选择


set_seed(1234)  # 设置为固定的随机种子，例如42

dataset = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/datasets_ton_iot_afterProcess.csv', encoding='utf-8',
                      low_memory=False)

dataset_7 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/little_class_data.csv', encoding='utf-8',
                        low_memory=False)



dataset_gen_7 = pd.read_csv(
    '/ton_iot/CVG/7_7783_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_6 = pd.read_csv(
    '/ton_iot/CVG/6_7439_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_5 = pd.read_csv(
    '/ton_iot/CVG/5_4912_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_4 = pd.read_csv(
    '/ton_iot/CVG/4_6804_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_2 = pd.read_csv(
    '/ton_iot/CVG/2_7308_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_1 = pd.read_csv(
    '/ton_iot/CVG/1_4149_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_0 = pd.read_csv(
    '/ton_iot/CVG/0_7397_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)


# print(df)

class Meself_dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        label = self.labels[item]
        return feature, label


# --------------------------------------------------------------------------------------------------------------------

# # # dataset = df.to_numpy()





dataset_gen_7 = dataset_gen_7.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_7.shape[0], 1), 7)
print(new_column)
# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_7 = np.hstack((dataset_gen_7, new_column))

dataset_gen_6 = dataset_gen_6.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_6.shape[0], 1), 6)

# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_6 = np.hstack((dataset_gen_6, new_column))

dataset_gen_5 = dataset_gen_5.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_5.shape[0], 1), 5)

# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_5 = np.hstack((dataset_gen_5, new_column))

dataset_gen_4 = dataset_gen_4.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_4.shape[0], 1), 4)

# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_4 = np.hstack((dataset_gen_4, new_column))

dataset_gen_2 = dataset_gen_2.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_2.shape[0], 1), 2)

# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_2 = np.hstack((dataset_gen_2, new_column))

dataset_gen_1 = dataset_gen_1.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_1.shape[0], 1), 1)

# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_1 = np.hstack((dataset_gen_1, new_column))

dataset_gen_0 = dataset_gen_0.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_0.shape[0], 1), 0)

# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_0 = np.hstack((dataset_gen_0, new_column))

# --------------------------------------------------------------------------------------------------------------------

train_data = []
test_data = []
val_data = []

# for label, group in dataset.groupby('class2'):
#     # 按比例划分
#     train_size = int(len(group) * 0.8)
#     test_size = int(len(group) * 0.1)
#
#     train, test_val = train_test_split(group, train_size=train_size, shuffle=True)
#     test, val = train_test_split(test_val, test_size=test_size, shuffle=True)
#
#     train_data.append(train)
#     test_data.append(test)
#     val_data.append(val)

for label, group in dataset.groupby('type'):
    # 按比例划分
    train_size = int(len(group) * 0.8)
    test_size = int(len(group) * 0.2)

    train, test = train_test_split(group, train_size=train_size, shuffle=True)
    # test, val = train_test_split(test_val, test_size=test_size, shuffle=True)

    train_data.append(train)
    test_data.append(test)
    # val_data.append(val)

# 合并数据
train_set = pd.concat(train_data).reset_index(drop=True)
test_set = pd.concat(test_data).reset_index(drop=True)
# val_set = pd.concat(val_data).reset_index(drop=True)


train_set = train_set.to_numpy()
print("train_set拼接前：", train_set.shape)
# --------------------------------------------------------------------------------------------------------------------
# 拼接行

train_set = np.concatenate((dataset_gen_7, train_set), axis=0)
train_set = np.concatenate((dataset_gen_6, train_set), axis=0)
train_set = np.concatenate((dataset_gen_5, train_set), axis=0)
train_set = np.concatenate((dataset_gen_4, train_set), axis=0)
train_set = np.concatenate((dataset_gen_2, train_set), axis=0)
train_set = np.concatenate((dataset_gen_1, train_set), axis=0)
train_set = np.concatenate((dataset_gen_0, train_set), axis=0)
# --------------------------------------------------------------------------------------------------------------------
print("train_set拼接后：", train_set.shape)
test_set = test_set.to_numpy()
# val_set = val_set.to_numpy()

print(train_set.shape)
print(type(train_set))

train_set = Meself_dataset(train_set[:, :-1], train_set[:, -1])
test_set = Meself_dataset(test_set[:, :-1], test_set[:, -1])
# val_set = Meself_dataset(val_set[:, :-1], val_set[:, -1])

batch_size = 64

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
# val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True)


train_features = []
train_labels = []

for featrue, label in train_loader:
    train_features.append(featrue)
    train_labels.append(label)
train_features = torch.cat(train_features)
train_labels = torch.cat(train_labels)

print(train_features.shape)
print(train_labels.shape)

train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

# # 决策树
# dt = DecisionTreeClassifier()
# dt.fit(train_features, train_labels)
#
# test_features = []
# test_labels = []
# for featrue, label in test_loader:
#     test_features.append(featrue)
#     test_labels.append(label)
# test_features = torch.cat(test_features)
# test_labels = torch.cat(test_labels)
#
# # 使用测试集进行预测
# y_pred = dt.predict(test_features)
#
# # 计算准确率
# accuracy = accuracy_score(test_labels, y_pred)
#
# print(f'测试集准确率: {accuracy:.2f}')
# # 输出每个类别的评估指标
#
# print(classification_report(test_labels, y_pred))


# 定义K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储每次的准确率和分类结果
accuracies = []
all_y_true = []
all_y_pred = []

for train_index, test_index in kf.split(train_features):
    X_train, X_test = train_features[train_index], train_features[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]

    # 创建并训练决策树模型
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # 输出分类报告
    print(classification_report(y_test, y_pred))

# # 计算平均准确率
# average_accuracy = np.mean(accuracies)
# print(f'Average Accuracy: {average_accuracy:.2f}')
#
# # 输出每个类别的准确率
# print("\nClassification Report:")
# print(classification_report(all_y_true, all_y_pred))
# 计算平均准确率
average_accuracy = np.mean(accuracies)
print(f'Average Accuracy: {average_accuracy:.2f}')

dataset_7 = dataset_7.to_numpy()
x_7 = torch.tensor(dataset_7, dtype=torch.float32)

y_7 = model.predict(x_7)
print(y_7)
