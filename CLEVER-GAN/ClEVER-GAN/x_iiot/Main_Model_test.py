import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import FlopCountAnalysis, parameter_count

# 查看当前 GPU 内存占用情况
def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也要设置随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内置随机数生成器的随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作的结果确定
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化算法选择


def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).float()
    return correct.sum() / len(labels)


set_seed(42)  # 设置为固定的随机种子，例如42

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/1000_vae_generated_data.csv', encoding='utf-8',
#                             low_memory=False)


data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/1000_vae_gan_contra_lossMethod_generated_data.csv',
                            encoding='utf-8',
                            low_memory=False)

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/1000_vae_gan_contra_generated_data.csv', encoding='utf-8',
#                             low_memory=False)

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/1000_GAN_generated_data.csv', encoding='utf-8',
#                             low_memory=False)


data_generate = data_generate.to_numpy()

data = pd.read_csv('D:/python/pythonProject/PHDfirstTest/datasets_xiiot_afterProcess.csv', encoding='utf-8',
                   low_memory=False)

data = data.to_numpy()

# # 指定要检查的列索引（从0开始）和要删除的值
# column_index = -2  # 第2列（索引为1）
# value_to_remove = 9
#
# # 使用布尔索引来筛选出不包含指定值的行
# filtered_data = data[data[:, column_index] != value_to_remove]

# 计算倒数第一列和倒数第三列的索引
index_last = -1  # 倒数第一列
index_third_last = -3  # 倒数第三列

# 使用 numpy.delete() 删除倒数第一列和倒数第三列
data = np.delete(data, [index_third_last, index_last], axis=1)

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

batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda")
print(torch.cuda.is_available())

# from VAE_test_Cnn import SimpleCNN
from MainModel import DepthwiseSeparableConv

# 实例化模型、定义损失函数和优化器
# 在训练开始时初始化 TensorBoard
# 定义超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 200
model = DepthwiseSeparableConv(batch_size).to(device)
writer = SummaryWriter()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# --------------------------------------------------------------------------------------------------------------------------------
# 计算参数量
# 创建输入张量
input_tensor = torch.randn(64, 1, 57).to(device)

# 计算 FLOPs
flop_analysis = FlopCountAnalysis(model, input_tensor)
flops = flop_analysis.total()

# 计算参数量
params = parameter_count(model)[""]

print(f"FLOPs: {flops}")
print(f"Params: {params}")

# --------------------------------------------------------------------------------------------------------------------------------


# 训练模型
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    train_loss = 0.0
    train_correct = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据加载到设备（GPU或CPU）
        data, target = data.to(device), target.to(device)
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

        train_loss += loss.item()
        train_correct += calculate_accuracy(output, target) * len(target)

        # # 每100个batch打印一次训练信息
        # if batch_idx % 10 == 99:
        #     print(
        #         f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / len(train_loader.dataset)

    # 在每个 epoch 中，记录训练的损失和准确率
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Accuracy/train', avg_train_acc, epoch)

    # 打印每个 epoch 结束时的损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}')

    # for param_group in optimizer.param_groups:
    #     current_lr = param_group['lr']
    # print(f'Current learning rate: {current_lr}')

    # 梯度范数（用于监控梯度爆炸或消失）
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f'Total Gradient Norm: {total_norm:.4f}')

    # 在每个 epoch 或者 batch 之后打印显存占用
    print_gpu_memory()

# 在训练结束时，关闭 TensorBoard
writer.close()

from sklearn.metrics import classification_report

# 假设你有训练好的模型，并且 `dataloader_test` 是你的测试数据加载器
model.eval()  # 切换到评估模式
all_preds = []
all_labels = []
correct = 0
total = 0

class_names = ['Normal', 'C&C', 'Exfiltration', 'Exploitation', 'Lateral _movement', 'RDOS', 'Reconnaissance',
               'Tampering', 'Weaponization', 'crypto-ransomware']

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.reshape(-1, 1, 57)
        data = data.float()  # 将数据类型转换为 float32
        target = target.long()

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        all_preds.extend(predicted.cpu().numpy())  # 将GPU上的张量移回CPU
        all_labels.extend(target.cpu().numpy())

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
# 生成分类报告
print(classification_report(all_labels, all_preds, target_names=class_names))
