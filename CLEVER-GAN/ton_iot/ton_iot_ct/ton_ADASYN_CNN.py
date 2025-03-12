from imblearn.over_sampling import SMOTE
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN


def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也要设置随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内置随机数生成器的随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作的结果确定
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化算法选择


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
set_seed(1234)  # 设置为固定的随机种子，例如42

dataset = pd.read_csv('D:/python/pythonProject/PHDfirstTest/ton_iot/datasets_ton_iot_afterProcess.csv',
                      encoding='utf-8',
                      low_memory=False)


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


train_data = []
test_data = []
val_data = []

for label, group in dataset.groupby('type'):
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
# print("train_set拼接前：", train_set.shape)
#
#
# print("train_set拼接后：", train_set.shape)
test_set = test_set.to_numpy()
# val_set = val_set.to_numpy()

print(train_set.shape)
print(type(train_set))

train_set = Meself_dataset(train_set[:, :-1], train_set[:, -1])
test_set = Meself_dataset(test_set[:, :-1], test_set[:, -1])
# val_set = Meself_dataset(val_set[:, :-1], val_set[:, -1])

batch_size = 64

print("type(train_set)", type(train_set))

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

# adasyn
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(train_features, train_labels)

# 输出结果
print(f"原始样本数量：{np.bincount(train_labels)}")
print(f"增加后的样本数量：{np.bincount(y_resampled.astype(int))}")  # 转换为整数类型

train_features = X_resampled
train_labels = y_resampled

print("type(train_features)", type(train_features))
train_labels = train_labels[:, np.newaxis]  # 或者使用 train_labels.reshape(-1, 1)
train_final = np.concatenate((train_features, train_labels), axis=1)
print(train_final.shape)

train_set = Meself_dataset(train_final[:, :-1], train_final[:, -1])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=9, kernel_size=20, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=24, kernel_size=20, stride=1, padding=1)
        self.fc1 = nn.Linear(312, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
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
