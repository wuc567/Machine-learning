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


set_seed(42)  # 设置为固定的随机种子，例如42

dataset = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/datasets_xiiot_afterProcess.csv', encoding='utf-8',
                      low_memory=False)

dataset_9 = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/little_class_data.csv', encoding='utf-8',
                        low_memory=False)

dataset_gen_9 = pd.read_csv(
    '/x_iiot/CVG/9_5960_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_8 = pd.read_csv(
    '/x_iiot/CVG/8_1069_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_7 = pd.read_csv(
    '/x_iiot/CVG/7_2632_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_6 = pd.read_csv(
    '/x_iiot/CVG/6_842_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_5 = pd.read_csv(
    '/x_iiot/CVG/5_1511_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_4 = pd.read_csv(
    '/x_iiot/CVG/4_713_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_3 = pd.read_csv(
    '/x_iiot/CVG/3_5154_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_1 = pd.read_csv(
    '/x_iiot/CVG/1_3970_best_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)
dataset_gen_0 = pd.read_csv(
    '/x_iiot/CVG/0_1939_best_vae_gan_contra_lossMethod_generated_data.csv',
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


dataset = dataset.drop(['class1', 'class3'], axis=1)
# --------------------------------------------------------------------------------------------------------------------

# # # dataset = df.to_numpy()

dataset_gen_9 = dataset_gen_9.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_9.shape[0], 1), 9)
print(new_column)
# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_9 = np.hstack((dataset_gen_9, new_column))

dataset_gen_8 = dataset_gen_8.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_8.shape[0], 1), 8)
print(new_column)
# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_8 = np.hstack((dataset_gen_8, new_column))

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

dataset_gen_3 = dataset_gen_3.to_numpy()
# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((dataset_gen_3.shape[0], 1), 3)

# 使用 numpy.hstack() 在原数组的右侧添加新列
dataset_gen_3 = np.hstack((dataset_gen_3, new_column))

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

for label, group in dataset.groupby('class2'):
    # 按比例划分
    train_size = int(len(group) * 0.6)
    test_size = int(len(group) * 0.4)

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
# train_set = np.concatenate((dataset_gen_9, train_set), axis=0)
# train_set = np.concatenate((dataset_gen_8, train_set), axis=0)
# train_set = np.concatenate((dataset_gen_7, train_set), axis=0)
# train_set = np.concatenate((dataset_gen_6, train_set), axis=0)
# train_set = np.concatenate((dataset_gen_5, train_set), axis=0)
# train_set = np.concatenate((dataset_gen_4, train_set), axis=0)
# train_set = np.concatenate((dataset_gen_3, train_set), axis=0)
# train_set = np.concatenate((dataset_gen_1, train_set), axis=0)
# train_set = np.concatenate((dataset_gen_0, train_set), axis=0)
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


# train_features = []
# train_labels = []
#
# for featrue, label in train_loader:
#     train_features.append(featrue)
#     train_labels.append(label)
# train_features = torch.cat(train_features)
# train_labels = torch.cat(train_labels)
#
# print(train_features.shape)
# print(train_labels.shape)
#
# train_features = torch.tensor(train_features, dtype=torch.float32)
# train_labels = torch.tensor(train_labels, dtype=torch.float32)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        #
        # # self.fc1 = nn.Linear(256, 64)
        # self.fc1 = nn.Linear(256, 10)
        # # self.fc2 = nn.Linear(32, 10)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=9, kernel_size=20, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=24, kernel_size=20, stride=1, padding=1)

        # self.fc1 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(24, 10)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool1d(x, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool1d(x, 2)
        # x = F.relu(self.conv3(x))
        # x = F.max_pool1d(x, 2)
        # x = F.relu(self.conv4(x))
        # x = F.max_pool1d(x, 2)
        # x = F.relu(self.conv5(x))
        # x = F.max_pool1d(x, 2)
        # # print(x.shape)
        #
        # x = x.view(64, -1)
        #
        # x = self.fc1(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        # print(x.shape)

        x = x.view(64, -1)

        x = self.fc1(x)

        return x


model = SimpleCNN()

num_epochs = 50
# 定义K折交叉验证

# 记录每折的准确率
fold_accuracies = []

# 存储每次的准确率和分类结果
accuracies = []
all_y_true = []
all_y_pred = []

# # 开始K折交叉验证
# for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
#     print(f'Fold {fold + 1}/{k_folds}')
#
#     # 初始化模型、损失函数和优化器
#     model = SimpleCNN().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#
#     # 训练模型
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for data in train_loader:
#             inputs, labels = data
#             inputs = inputs.float().to(device)
#             labels = labels.long().to(device)  # 将标签转换为 Long 类型
#             inputs = inputs.reshape(batch_size, 1, -1)
#             # print("input：", inputs.shape)
#             optimizer.zero_grad()
#             # 前向传播
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             # 反向传播和优化
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#         # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
#
#     # 验证模型
#     model.eval()
#     correct = 0
#     total = 0
#     all_labels = []
#     all_preds = []
#     with torch.no_grad():
#         for data in test_loader:
#             inputs, labels = data
#             inputs = inputs.float().to(device)
#             labels = labels.long().to(device)  # 将标签转换为 Long 类型
#             inputs = inputs.reshape(batch_size, 1, -1)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())
#
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#     fold_accuracies.append(accuracy)
#     # 计算每个类别的 precision, recall, f1-score
#     report = classification_report(all_labels, all_preds, digits=4)
#     print(f'Fold {fold + 1} Classification Report:\n', report)
#     print(f'Fold {fold + 1} Accuracy: {accuracy:.2f}%')
#
# # 打印每折的平均准确率
# print(f'K-Fold Cross Validation Results: {k_folds} Folds')
# print(f'Average Accuracy: {sum(fold_accuracies) / len(fold_accuracies):.2f}%')
# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)  # 将标签转换为 Long 类型
        inputs = inputs.reshape(batch_size, 1, -1)
        # print("input：", inputs.shape)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 验证模型
model.eval()
correct = 0
total = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)  # 将标签转换为 Long 类型
        inputs = inputs.reshape(batch_size, 1, -1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
fold_accuracies.append(accuracy)
# 计算每个类别的 precision, recall, f1-score
report = classification_report(all_labels, all_preds, digits=4)
print(f'Classification Report:\n', report)
print(f'Accuracy: {accuracy:.2f}%')
