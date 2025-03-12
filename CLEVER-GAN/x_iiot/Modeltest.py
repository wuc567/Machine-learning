import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random

def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也要设置随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内置随机数生成器的随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作的结果确定
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化算法选择



set_seed(42)  # 设置为固定的随机种子，例如42

# data_generate = pd.read_csv('/x_iiot/vae_generated_data.csv', encoding='utf-8',
#                             low_memory=False)

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/best_vae_generated_data.csv', encoding='utf-8',
#                             low_memory=False)

data_generate = pd.read_csv('/x_iiot/CVG/9_5960_best_vae_gan_contra_lossMethod_generated_data.csv', encoding='utf-8',
                            low_memory=False)

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/vae_gan_contra_generated_data.csv', encoding='utf-8',
#                             low_memory=False)

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/GAN_generated_data.csv', encoding='utf-8',
#                             low_memory=False)


data_generate = data_generate.to_numpy()

data = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/datasets_xiiot_afterProcess.csv', encoding='utf-8',
                   low_memory=False)

data = data.to_numpy()

# 指定要检查的列索引（从0开始）和要删除的值
column_index = -2  # 第2列（索引为1）
value_to_remove = 9

# 使用布尔索引来筛选出不包含指定值的行
filtered_data = data[data[:, column_index] != value_to_remove]

# 计算倒数第一列和倒数第三列的索引
index_last = -1  # 倒数第一列
index_third_last = -3  # 倒数第三列

# 使用 numpy.delete() 删除倒数第一列和倒数第三列
data = np.delete(filtered_data, [index_third_last, index_last], axis=1)

# 创建一个元素全为9的新列，行数与原数组匹配
new_column = np.full((data_generate.shape[0], 1), 9)

# 使用 numpy.hstack() 在原数组的右侧添加新列
data_generate = np.hstack((data_generate, new_column))

#
# print(data.shape)
# print(data_generate.shape)


# 纵向拼接
data_all = np.concatenate((data, data_generate), axis=0)

print(data.shape)
print(data_generate.shape)
print(data_all.shape)

feature = data_all[:, 0:-1]
lable = data_all[:, -1]

print(feature)
print(lable)

# 划分验证集和测试集

all_data = np.concatenate([feature, lable.reshape(-1, 1)], axis=1)

# 转化为dataset
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


class All_dataset(Dataset):
    def __init__(self):
        data = all_data
        self._x = torch.from_numpy(data[:, :-1])  # 线性回归和DNN不用升维
        self.label = torch.from_numpy(data[:, -1])
        self.len = len(data)

    def __getitem__(self, item):
        return self._x[item], self.label[item]

    def __len__(self):
        return self.len


# all_dataset = All_dataset()
from torch.utils.data import random_split

dataset = All_dataset()  # 原始数据集
train_ratio = 0.8  # 训练集的比例，例如取80%
val_ratio = 0.1  # 验证集的比例，例如取10%
test_ratio = 0.1  # 测试集的比例，例如取10%

train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 16

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda")
print(torch.cuda.is_available())

from VAE_test_Cnn import SimpleCNN

# 定义超参数
batch_size = 16
learning_rate = 0.001
num_epochs = 10
# 实例化模型、定义损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据加载到设备（GPU或CPU）
        data, target = data, target
        data = data.reshape(-1, 1, 57)
        data = data.float()  # 将数据类型转换为 float32
        # print(data.shape)
        target = target.long()
        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每100个batch打印一次训练信息
        if batch_idx % 100 == 99:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# 测试模型
model.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data, target
        data = data.reshape(-1, 1, 57)
        data = data.float()  # 将数据类型转换为 float32
        target = target.long()

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

data_text = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/datasets_xiiot_afterProcess.csv', encoding='utf-8',
                        low_memory=False)

data_text = data_text.to_numpy()
# 假设我们要获取倒数第二列中的值等于 9 的所有行
target_value = 9

# 使用布尔索引获取倒数第二列等于 target_value 的行
text_Result = data_text[data_text[:, -2] == target_value]

# # 使用 numpy.delete() 删除倒数第一列和倒数第三列
# text_Result = np.delete(text_Result, [index_third_last, index_last], axis=1)
text_Result = text_Result[:, :-3]
text_Result = text_Result[:16, :]


print(text_Result)
print(text_Result.shape)
test_tensor = torch.tensor(text_Result, dtype=torch.float32)


test_tensor = test_tensor.reshape(-1, 1, 57)

# 禁用梯度计算，以便节省内存并提高速度
with torch.no_grad():
    # 输入测试数据进行预测
    output = model(test_tensor)

# 如果是分类任务，使用 argmax 获取预测的类别
predicted_class = torch.argmax(output, dim=1)
print("最后结果：",predicted_class)
