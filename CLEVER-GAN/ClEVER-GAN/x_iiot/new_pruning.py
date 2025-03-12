import os
import time
from copy import deepcopy
import torchprofile
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils.prune as prune
from sklearn.metrics import classification_report


def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也要设置随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内置随机数生成器的随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作的结果确定
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化算法选择


set_seed(42)  # 设置为固定的随机种子，例如42


# 生成器，用于生成每层的裁剪比例
class PruningGenerator(nn.Module):
    def __init__(self, noise_dim, num_layers):
        super(PruningGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_layers),
            nn.Sigmoid()  # 生成 [0, 1] 之间的裁剪比例
        )

    def forward(self, z):
        return self.fc(z)


def gen_get_loss(amounts):
    # amounts_tensor = torch.tensor(amounts)
    # return -torch.log(1 + torch.sum(amounts))
    return torch.exp(-torch.sum(amounts))


def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).float()
    return correct.sum() / len(labels)


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# device = torch.device("cpu")

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/1000_vae_generated_data.csv', encoding='utf-8',
#                             low_memory=False)


data_generate = pd.read_csv(
    'D:/python/pythonProject/PHDfirstTest/x_iiot/x_iiot_1000_vae_gan_contra_lossMethod_generated_data.csv',
    encoding='utf-8',
    low_memory=False)

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/1000_vae_gan_contra_generated_data.csv', encoding='utf-8',
#                             low_memory=False)

# data_generate = pd.read_csv('D:/python/pythonProject/PHDfirstTest/1000_GAN_generated_data.csv', encoding='utf-8',
#                             low_memory=False)


data_generate = data_generate.to_numpy()

data = pd.read_csv('D:/python/pythonProject/PHDfirstTest/x_iiot/datasets_xiiot_afterProcess.csv', encoding='utf-8',
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

print(torch.cuda.is_available())

from MainModel import DepthwiseSeparableConv

# # 查看生成的掩码形状
# for mask in masks:
#     print(f"Generated mask shape: {mask.shape}")


# 定义超参数
z_dim = 50
batch_size = 64
main_learning_rate = 0.001
generator_learning_rate = 0.000001
num_epochs = 10
writer = SummaryWriter()

main_model = DepthwiseSeparableConv(batch_size).to(device)

# main_model = DepthwiseSeparableConv(batch_size)
criterion = nn.CrossEntropyLoss()
main_model_optimizer = optim.Adam(main_model.parameters(), lr=main_learning_rate)

noise_dim = 50
num_layers = 10
generator = PruningGenerator(noise_dim, num_layers).to(device)
generator_optimizer = optim.Adam(generator.parameters(), lr=generator_learning_rate)

scheduler = ReduceLROnPlateau(generator_optimizer, mode='min', factor=0.1, patience=2)

main_model.train()  # 设置🐖模型为训练模式
generator.train()

# --------------------------------------------------------------------------------------训练开始-------------------------------------------------------------------------------------------

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据加载到设备（GPU或CPU）
        data, target = data.to(device), target.to(device)
        data = data.reshape(-1, 1, 57)
        data = data.float()  # 将数据类型转换为 float32
        target = target.long()

        # --------------------------------------------------训练主模型（使用生成器生成掩码并传入主模型）-----------------------------------------------------------------

        # noise = torch.ones(1, z_dim).to(device)  # 假设噪声向量大小为 50
        noise = torch.randn(1, z_dim).to(device)  # 假设噪声向量大小为 50
        amounts = generator(noise)
        amounts2 = amounts.squeeze().tolist()

        with torch.no_grad():
            prune.l1_unstructured(main_model.depthwise1, name='weight', amount=amounts2[0])
            prune.l1_unstructured(main_model.pointwise1, name='weight', amount=amounts2[1])
            prune.l1_unstructured(main_model.depthwise2, name='weight', amount=amounts2[2])
            prune.l1_unstructured(main_model.pointwise2, name='weight', amount=amounts2[3])
            prune.l1_unstructured(main_model.depthwise3, name='weight', amount=amounts2[4])
            prune.l1_unstructured(main_model.pointwise3, name='weight', amount=amounts2[5])
            prune.l1_unstructured(main_model.fc1, name='weight', amount=amounts2[6])
            prune.l1_unstructured(main_model.fc2, name='weight', amount=amounts2[7])
            prune.l1_unstructured(main_model.fc3, name='weight', amount=amounts2[8])
            prune.l1_unstructured(main_model.fc4, name='weight', amount=amounts2[9])

        # 移除已经被裁剪的权重
        prune.remove(main_model.depthwise1, 'weight')
        prune.remove(main_model.pointwise1, 'weight')
        prune.remove(main_model.depthwise2, 'weight')
        prune.remove(main_model.pointwise2, 'weight')
        prune.remove(main_model.depthwise3, 'weight')
        prune.remove(main_model.pointwise3, 'weight')
        prune.remove(main_model.fc1, 'weight')
        prune.remove(main_model.fc2, 'weight')
        prune.remove(main_model.fc3, 'weight')
        prune.remove(main_model.fc4, 'weight')

        output = main_model(data)
        main_model_loss = criterion(output, target)
        main_model_optimizer.zero_grad()
        main_model_loss.backward()
        main_model_optimizer.step()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------训练生成器--------------------------------------------------------------------------

        generator_loss = gen_get_loss(amounts)
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------

        train_loss += main_model_loss.item()
        train_correct += calculate_accuracy(output, target) * len(target)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / len(train_loader.dataset)

    prev_model_loss = avg_train_loss  # 记录当前模型的损失

    # 在每个 epoch 中，记录训练的损失和准确率
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Accuracy/train', avg_train_acc, epoch)

    scheduler.step(avg_train_loss)
    # 获取当前学习率
    current_lr = scheduler.get_last_lr()[0]
    # 打印每个 epoch 结束时的损失
    # 打印训练过程中的损失
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], generator Loss: {generator_loss.item():.4f}, model Loss: {main_model_loss.item():.4f}, model Accuracy: {avg_train_acc:.4f},generator learning rate:{current_lr:.20f}")


# --------------------------------------------------------------------------------------训练结束-----------------------------------------------------------------------------------

def count_nonzero_parameters(model):
    total_nonzero_params = sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)
    return total_nonzero_params


main_model.eval()
generator.eval()
# noise = torch.ones(1, z_dim).to(device)  # 假设噪声向量大小为 50
noise = torch.randn(1, z_dim).to(device)  # 假设噪声向量大小为 50
amounts_generator = generator(noise).squeeze().tolist()
print(amounts_generator)

original_model = deepcopy(main_model)

# print(f"剪枝前非零参数数量: {count_nonzero_parameters(original_model)}")


with torch.no_grad():
    prune.l1_unstructured(main_model.depthwise1, name='weight', amount=amounts_generator[0])
    prune.l1_unstructured(main_model.pointwise1, name='weight', amount=amounts_generator[1])
    prune.l1_unstructured(main_model.depthwise2, name='weight', amount=amounts_generator[2])
    prune.l1_unstructured(main_model.pointwise2, name='weight', amount=amounts_generator[3])
    prune.l1_unstructured(main_model.depthwise3, name='weight', amount=amounts_generator[4])
    prune.l1_unstructured(main_model.pointwise3, name='weight', amount=amounts_generator[5])
    prune.l1_unstructured(main_model.fc1, name='weight', amount=amounts_generator[6])
    prune.l1_unstructured(main_model.fc2, name='weight', amount=amounts_generator[7])
    prune.l1_unstructured(main_model.fc3, name='weight', amount=amounts_generator[8])
    prune.l1_unstructured(main_model.fc4, name='weight', amount=amounts_generator[9])

# 移除已经被裁剪的权重
prune.remove(main_model.depthwise1, 'weight')
prune.remove(main_model.pointwise1, 'weight')
prune.remove(main_model.depthwise2, 'weight')
prune.remove(main_model.pointwise2, 'weight')
prune.remove(main_model.depthwise3, 'weight')
prune.remove(main_model.pointwise3, 'weight')
prune.remove(main_model.fc1, 'weight')
prune.remove(main_model.fc2, 'weight')
prune.remove(main_model.fc3, 'weight')
prune.remove(main_model.fc4, 'weight')

pruned_model = deepcopy(main_model)

data = torch.randn(64, 1, 57).to(device)  # 根据你的数据形状调整
num = 10000

# 测试剪枝前模型的速度
original_model.eval()
with torch.no_grad():
    start_time = time.time()
    for _ in range(num):  # 重复执行num次，取平均值
        output = original_model(data)
    end_time = time.time()
    prune_time = (end_time - start_time) / num
    print(f"剪枝前模型的平均前向传播时间: {prune_time:.6f} 秒")

pruned_model.eval()
with torch.no_grad():
    start_time = time.time()
    for _ in range(num):
        output = pruned_model(data)
    end_time = time.time()
    post_prune_time = (end_time - start_time) / num
    print(f"剪枝后模型的平均前向传播时间: {post_prune_time:.6f} 秒")

print(f"剪枝前非零参数数量: {count_nonzero_parameters(original_model)}")

print(f"剪枝后非零参数数量: {count_nonzero_parameters(pruned_model)}")

total_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
print(f"剪枝前总参数数量: {total_params}")

total_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
print(f"剪枝后总参数数量: {total_params}")

dummy_input = torch.randn(64, 1, 57).to(device)  # 根据你的输入形状进行调整
flops1 = torchprofile.profile_macs(original_model, dummy_input)
print(f"原始模型FLOPs: {flops1}")

dummy_input = torch.randn(64, 1, 57).to(device)  # 根据你的输入形状进行调整
flops2 = torchprofile.profile_macs(pruned_model, dummy_input)
print(f"剪枝后模型FLOPs: {flops2}")

all_preds = []
all_labels = []
correct = 0
total = 0

class_names = ['Normal', 'C&C', 'Exfiltration', 'Exploitation', 'Lateral _movement', 'RDOS', 'Reconnaissance',
               'Tampering', 'Weaponization', 'crypto-ransomware']

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # data, target = data, target
        data = data.reshape(-1, 1, 57)
        data = data.float()  # 将数据类型转换为 float32
        target = target.long()

        outputs = main_model(data)
        # outputs = pruned_model(data)

        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        all_preds.extend(predicted.cpu().numpy())  # 将GPU上的张量移回CPU
        all_labels.extend(target.cpu().numpy())

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
# 生成分类报告
print(classification_report(all_labels, all_preds, target_names=class_names))

# # 保存剪枝前的模型
# torch.save(original_model.state_dict(), 'x_iiot_main_model_before_pruning.pth')
#
# # 将剪枝后模型的参数转换为稀疏格式并保存
# for name, param in pruned_model.named_parameters():
#     if param is not None:
#         pruned_model.state_dict()[name] = param.to_sparse()
#
# torch.save(pruned_model.state_dict(), 'x_iiot_pruned_model_sparse.pth')
#
# # 查看文件大小（字节为单位）
# before_size = os.path.getsize('x_iiot_main_model_before_pruning.pth')
# after_size = os.path.getsize('x_iiot_pruned_model_sparse.pth')
#
# print(f"剪枝前模型文件大小: {before_size / (1024 * 1024):.2f} MB")
# print(f"剪枝后模型文件大小: {after_size / (1024 * 1024):.2f} MB")


dense_weight1 = original_model.depthwise1.weight.data
dense_weight2 = original_model.pointwise1.weight.data
dense_weight3 = original_model.depthwise2.weight.data
dense_weight4 = original_model.pointwise2.weight.data
dense_weight5 = original_model.depthwise3.weight.data
dense_weight6 = original_model.pointwise3.weight.data
dense_weight7 = original_model.fc1.weight.data
dense_weight8 = original_model.fc2.weight.data
dense_weight9 = original_model.fc3.weight.data
dense_weight10 = original_model.fc4.weight.data
print(dense_weight1)
print(dense_weight2)
print(dense_weight3)

# 计算非零元素的数量
non_zero_count1 = torch.count_nonzero(dense_weight1)
non_zero_count2 = torch.count_nonzero(dense_weight2)
non_zero_count3 = torch.count_nonzero(dense_weight3)
non_zero_count4 = torch.count_nonzero(dense_weight4)
non_zero_count5 = torch.count_nonzero(dense_weight5)
non_zero_count6 = torch.count_nonzero(dense_weight6)
non_zero_count7 = torch.count_nonzero(dense_weight7)
non_zero_count8 = torch.count_nonzero(dense_weight8)
non_zero_count9 = torch.count_nonzero(dense_weight9)
non_zero_count10 = torch.count_nonzero(dense_weight10)
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count1}")
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count2}")
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count3}")
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count4}")
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count5}")
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count6}")
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count7}")
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count8}")
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count9}")
print(f"剪枝前稀疏矩阵的非零元素数量: {non_zero_count10}")


print(f"dense_weight1.nelement: {dense_weight1.nelement()}")
print(f"dense_weight2.nelement: {dense_weight2.nelement()}")
print(f"dense_weight3.nelement: {dense_weight3.nelement()}")

# 计算密集矩阵的存储空间
dense_size1 = dense_weight1.nelement() * dense_weight1.element_size()  # 元素数量 * 每个元素的字节数
dense_size2 = dense_weight2.nelement() * dense_weight2.element_size()  # 元素数量 * 每个元素的字节数
dense_size3 = dense_weight3.nelement() * dense_weight3.element_size()  # 元素数量 * 每个元素的字节数
print(dense_size3)
dense_size4 = dense_weight4.nelement() * dense_weight4.element_size()  # 元素数量 * 每个元素的字节数
dense_size5 = dense_weight5.nelement() * dense_weight5.element_size()  # 元素数量 * 每个元素的字节数
dense_size6 = dense_weight6.nelement() * dense_weight6.element_size()  # 元素数量 * 每个元素的字节数
dense_size7 = dense_weight7.nelement() * dense_weight7.element_size()  # 元素数量 * 每个元素的字节数
dense_size8 = dense_weight8.nelement() * dense_weight8.element_size()  # 元素数量 * 每个元素的字节数
dense_size9 = dense_weight9.nelement() * dense_weight9.element_size()  # 元素数量 * 每个元素的字节数
dense_size10 = dense_weight10.nelement() * dense_weight10.element_size()  # 元素数量 * 每个元素的字节数

dense_size_all = dense_size1 + dense_size2 + dense_size3 + dense_size4 + dense_size5 + dense_size6 + dense_size7 + dense_size8 + dense_size9 + dense_size10
print(f"稠密矩阵占用存储空间: {dense_size_all / (1024 * 1024):.2f} MB")






dense_weight1 = pruned_model.depthwise1.weight.data
dense_weight2 = pruned_model.pointwise1.weight.data
dense_weight3 = pruned_model.depthwise2.weight.data
dense_weight4 = pruned_model.pointwise2.weight.data
dense_weight5 = pruned_model.depthwise3.weight.data
dense_weight6 = pruned_model.pointwise3.weight.data
dense_weight7 = pruned_model.fc1.weight.data
dense_weight8 = pruned_model.fc2.weight.data
dense_weight9 = pruned_model.fc3.weight.data
dense_weight10 = pruned_model.fc4.weight.data

print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print(dense_weight1)
print(dense_weight2)
print(dense_weight3)
# 计算非零元素的数量
non_zero_count1 = torch.count_nonzero(dense_weight1)
non_zero_count2 = torch.count_nonzero(dense_weight2)
non_zero_count3 = torch.count_nonzero(dense_weight3)
non_zero_count4 = torch.count_nonzero(dense_weight4)
non_zero_count5 = torch.count_nonzero(dense_weight5)
non_zero_count6 = torch.count_nonzero(dense_weight6)
non_zero_count7 = torch.count_nonzero(dense_weight7)
non_zero_count8 = torch.count_nonzero(dense_weight8)
non_zero_count9 = torch.count_nonzero(dense_weight9)
non_zero_count10 = torch.count_nonzero(dense_weight10)
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count1}")
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count2}")
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count3}")
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count4}")
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count5}")
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count6}")
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count7}")
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count8}")
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count9}")
print(f"剪枝后稀疏矩阵的非零元素数量: {non_zero_count10}")


# 将权重转换为稀疏格式 (COO 格式)
sparse_weight1 = dense_weight1.to_sparse()
print(f"sparse_weight1：{sparse_weight1}")

sparse_values_size1 = sparse_weight1._values().nelement() * sparse_weight1._values().element_size()  # 非零值的存储空间
sparse_indices_size1 = sparse_weight1._indices().nelement() * sparse_weight1._indices().element_size()  # 非零值的索引的存储空间
sparse_total_size1 = sparse_values_size1 + sparse_indices_size1

sparse_weight2 = dense_weight2.to_sparse()
sparse_values_size2 = sparse_weight2._values().nelement() * sparse_weight2._values().element_size()  # 非零值的存储空间
sparse_indices_size2 = sparse_weight2._indices().nelement() * sparse_weight2._indices().element_size()  # 非零值的索引的存储空间
sparse_total_size2 = sparse_values_size2 + sparse_indices_size2
print(f"sparse_weight2：{sparse_weight2}")

sparse_weight3 = dense_weight3.to_sparse()
sparse_values_size3 = sparse_weight3._values().nelement() * sparse_weight3._values().element_size()  # 非零值的存储空间
sparse_indices_size3 = sparse_weight3._indices().nelement() * sparse_weight3._indices().element_size()  # 非零值的索引的存储空间
print(f"sparse_weight3：{sparse_weight3}")
print(f"sparse_weight3._indices：{sparse_weight3._indices()}")


print(sparse_values_size3)
print(sparse_indices_size3)


sparse_total_size3 = sparse_values_size3 + sparse_indices_size3

sparse_weight4 = dense_weight4.to_sparse()
sparse_values_size4 = sparse_weight4._values().nelement() * sparse_weight4._values().element_size()  # 非零值的存储空间
sparse_indices_size4 = sparse_weight4._indices().nelement() * sparse_weight4._indices().element_size()  # 非零值的索引的存储空间
sparse_total_size4 = sparse_values_size4 + sparse_indices_size4

sparse_weight5 = dense_weight5.to_sparse()
sparse_values_size5 = sparse_weight5._values().nelement() * sparse_weight5._values().element_size()  # 非零值的存储空间
sparse_indices_size5 = sparse_weight5._indices().nelement() * sparse_weight5._indices().element_size()  # 非零值的索引的存储空间
sparse_total_size5 = sparse_values_size5 + sparse_indices_size5

sparse_weight6 = dense_weight6.to_sparse()
sparse_values_size6 = sparse_weight6._values().nelement() * sparse_weight6._values().element_size()  # 非零值的存储空间
sparse_indices_size6 = sparse_weight6._indices().nelement() * sparse_weight6._indices().element_size()  # 非零值的索引的存储空间
sparse_total_size6 = sparse_values_size6 + sparse_indices_size6

sparse_weight7 = dense_weight7.to_sparse()
sparse_values_size7 = sparse_weight7._values().nelement() * sparse_weight7._values().element_size()  # 非零值的存储空间
sparse_indices_size7 = sparse_weight7._indices().nelement() * sparse_weight7._indices().element_size()  # 非零值的索引的存储空间
sparse_total_size7 = sparse_values_size7 + sparse_indices_size7

sparse_weight8 = dense_weight8.to_sparse()
sparse_values_size8 = sparse_weight8._values().nelement() * sparse_weight8._values().element_size()  # 非零值的存储空间
sparse_indices_size8 = sparse_weight8._indices().nelement() * sparse_weight8._indices().element_size()  # 非零值的索引的存储空间
sparse_total_size8 = sparse_values_size8 + sparse_indices_size8

sparse_weight9 = dense_weight9.to_sparse()
sparse_values_size9 = sparse_weight9._values().nelement() * sparse_weight9._values().element_size()  # 非零值的存储空间
sparse_indices_size9 = sparse_weight9._indices().nelement() * sparse_weight9._indices().element_size()  # 非零值的索引的存储空间
sparse_total_size9 = sparse_values_size9 + sparse_indices_size9

sparse_weight10 = dense_weight10.to_sparse()
sparse_values_size10 = sparse_weight10._values().nelement() * sparse_weight10._values().element_size()  # 非零值的存储空间
sparse_indices_size10 = sparse_weight10._indices().nelement() * sparse_weight10._indices().element_size()  # 非零值的索引的存储空间
sparse_total_size10 = sparse_values_size10 + sparse_indices_size10

sparse_total_size_all = sparse_total_size1 + sparse_total_size2 + sparse_total_size3 + sparse_total_size4 + sparse_total_size5 + sparse_total_size6 + sparse_total_size7 + sparse_total_size8 + sparse_total_size9 + sparse_total_size10

print(f"稀疏矩阵占用存储空间: {sparse_total_size_all / (1024 * 1024):.2f} MB")
