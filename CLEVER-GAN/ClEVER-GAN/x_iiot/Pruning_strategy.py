import time
from copy import deepcopy

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.optim.lr_scheduler import ReduceLROnPlateau


def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，也要设置随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内置随机数生成器的随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作的结果确定
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化算法选择


set_seed(42)  # 设置为固定的随机种子，例如42


def gen_loss_l1(masks, lambda_prune=1e-6):
    # 剪枝度量损失函数（基于 L1 正则化）
    loss = 0
    for mask in masks:
        loss += torch.sum(torch.abs(mask))  # L1 正则化，鼓励稀疏性
    return lambda_prune * loss


def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).float()
    return correct.sum() / len(labels)


class MaskGenerator(nn.Module):
    def __init__(self, z_dim, weight_shapes):
        super(MaskGenerator, self).__init__()
        # 生成器的不同层，输出对应权重形状的掩码
        self.fc1 = nn.Linear(z_dim, weight_shapes[0][0] * weight_shapes[0][1] * weight_shapes[0][2])  # 对应权重 [1, 1, 3]
        self.fc2 = nn.Linear(z_dim, weight_shapes[1][0] * weight_shapes[1][1] * weight_shapes[1][2])  # 对应权重 [16, 1, 1]
        self.fc3 = nn.Linear(z_dim, weight_shapes[2][0] * weight_shapes[2][1] * weight_shapes[2][2])  # 对应权重 [16, 1, 3]
        self.fc4 = nn.Linear(z_dim, weight_shapes[3][0] * weight_shapes[3][1] * weight_shapes[3][2])  # 对应权重 [32, 16, 1]
        self.fc5 = nn.Linear(z_dim, weight_shapes[4][0] * weight_shapes[4][1] * weight_shapes[4][2])  # 对应权重 [32, 1, 3]
        self.fc6 = nn.Linear(z_dim, weight_shapes[5][0] * weight_shapes[5][1] * weight_shapes[5][2])  # 对应权重 [64, 32, 1]
        self.fc7 = nn.Linear(z_dim, weight_shapes[6][0] * weight_shapes[6][1])  # 对应权重 [100, 192]
        self.fc8 = nn.Linear(z_dim, weight_shapes[7][0] * weight_shapes[7][1])  # 对应权重 [50, 100]
        self.fc9 = nn.Linear(z_dim, weight_shapes[8][0] * weight_shapes[8][1])  # 对应权重 [20, 50]
        self.fc10 = nn.Linear(z_dim, weight_shapes[9][0] * weight_shapes[9][1])  # 对应权重 [10, 20]

    def forward(self, z):
        # 每一层生成对应形状的掩码
        mask1 = torch.sigmoid(self.fc1(z)).view(weight_shapes[0][0], weight_shapes[0][1],
                                                weight_shapes[0][2])  # 二值化前的连续值
        mask2 = torch.sigmoid(self.fc2(z)).view(weight_shapes[1][0], weight_shapes[1][1], weight_shapes[1][2])
        mask3 = torch.sigmoid(self.fc3(z)).view(weight_shapes[2][0], weight_shapes[2][1], weight_shapes[2][2])
        mask4 = torch.sigmoid(self.fc4(z)).view(weight_shapes[3][0], weight_shapes[3][1], weight_shapes[3][2])
        mask5 = torch.sigmoid(self.fc5(z)).view(weight_shapes[4][0], weight_shapes[4][1], weight_shapes[4][2])
        mask6 = torch.sigmoid(self.fc6(z)).view(weight_shapes[5][0], weight_shapes[5][1], weight_shapes[5][2])
        mask7 = torch.sigmoid(self.fc7(z)).view(weight_shapes[6][0], weight_shapes[6][1])
        mask8 = torch.sigmoid(self.fc8(z)).view(weight_shapes[7][0], weight_shapes[7][1])
        mask9 = torch.sigmoid(self.fc9(z)).view(weight_shapes[8][0], weight_shapes[8][1])
        mask10 = torch.sigmoid(self.fc10(z)).view(weight_shapes[9][0], weight_shapes[9][1])

        # 对掩码进行二值化操作，生成 0/1 掩码
        # 二值化处理，并用 detach 保持反向传播能力
        mask1_binary = (mask1 > 0.5).float() + (mask1 - mask1.detach())
        mask2_binary = (mask2 > 0.5).float() + (mask2 - mask2.detach())
        mask3_binary = (mask3 > 0.5).float() + (mask3 - mask3.detach())
        mask4_binary = (mask4 > 0.5).float() + (mask4 - mask4.detach())
        mask5_binary = (mask5 > 0.5).float() + (mask5 - mask5.detach())
        mask6_binary = (mask6 > 0.5).float() + (mask6 - mask6.detach())
        mask7_binary = (mask7 > 0.5).float() + (mask7 - mask7.detach())
        mask8_binary = (mask8 > 0.5).float() + (mask8 - mask8.detach())
        mask9_binary = (mask9 > 0.5).float() + (mask9 - mask9.detach())
        mask10_binary = (mask10 > 0.5).float() + (mask10 - mask10.detach())

        return [mask1_binary, mask2_binary, mask3_binary, mask4_binary, mask5_binary, mask6_binary, mask7_binary,
                mask8_binary, mask9_binary, mask10_binary]


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

device = torch.device("cuda")
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
# generator_learning_rate = 0.001
num_epochs = 10
writer = SummaryWriter()

main_model = DepthwiseSeparableConv(batch_size).to(device)

# main_model = DepthwiseSeparableConv(batch_size)
criterion = nn.CrossEntropyLoss()
main_model_optimizer = optim.Adam(main_model.parameters(), lr=main_learning_rate)

# 获取权重的形状列表
weight_shapes = [param.shape for name, param in main_model.named_parameters() if 'weight' in name]
print(weight_shapes)

generator = MaskGenerator(z_dim, weight_shapes).to(device)
# generator = MaskGenerator(z_dim, weight_shapes)
generator_optimizer = optim.Adam(generator.parameters(), lr=generator_learning_rate)


# # 初始化生成器，生成与权重形状相匹配的掩码
# noise = torch.randn(1, z_dim)  # 假设噪声向量大小为 50
# mask_generator = MaskGenerator(z_dim, weight_shapes)
# masks = mask_generator(noise)
# 训练模型

# --------------------------------------------------------------------------------------------------------------------------------
# 计算参数量
# 创建输入张量
# input_tensor = torch.randn(64, 1, 57).to(device)
# input_tensor = torch.randn(64, 1, 57)
#
# # 计算 FLOPs
# flop_analysis = FlopCountAnalysis(main_model, input_tensor)
# flops = flop_analysis.total()
#
# # 计算参数量
# params = parameter_count(main_model)[""]
#
# print(f"FLOPs: {flops}")
# print(f"Params: {params}")
# 打印原始模型参数数量
def count_params(model):
    return sum(p.numel() for p in model.parameters())


print(f"Original parameter count: {count_params(main_model)}")

# --------------------------------------------------------------------------------------------------------------------------------

scheduler = ReduceLROnPlateau(generator_optimizer, mode='min', factor=0.1, patience=2)

main_model.train()  # 设置🐖模型为训练模式
generator.train()

# 初始化 L1 损失权重系数
lambda_l1 = 1.0
prev_model_loss = float('inf')  # 记录前一个 epoch 的主模型损失
patience = 2  # 设定一个耐心值，允许在几个 epoch 内模型损失有小幅波动
alpha = 1.0

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据加载到设备（GPU或CPU）
        data, target = data.to(device), target.to(device)
        # data, target = data, target
        data = data.reshape(-1, 1, 57)
        data = data.float()  # 将数据类型转换为 float32
        # print(data.shape)
        target = target.long()

        # --------------------------------------------------训练主模型（使用生成器生成掩码并传入主模型）-----------------------------------------------------------------

        noise = torch.ones(1, z_dim).to(device)  # 假设噪声向量大小为 50
        # noise = torch.ones(1, z_dim)  # 假设噪声向量大小为 50
        masks = generator(noise)

        # 应用掩码
        with torch.no_grad():
            main_model.depthwise1.weight *= masks[0]
            main_model.pointwise1.weight *= masks[1]
            main_model.depthwise2.weight *= masks[2]
            main_model.pointwise2.weight *= masks[3]
            main_model.depthwise3.weight *= masks[4]
            main_model.pointwise3.weight *= masks[5]
            main_model.fc1.weight *= masks[6]
            main_model.fc2.weight *= masks[7]
            main_model.fc3.weight *= masks[8]
            main_model.fc4.weight *= masks[9]

        output = main_model(data)
        main_model_loss = criterion(output, target)
        main_model_optimizer.zero_grad()
        # main_model_loss.backward(retain_graph=True)
        main_model_loss.backward()
        main_model_optimizer.step()

        # main_model_optimizer.zero_grad()
        # main_model_loss.backward()
        # main_model_optimizer.step()

        # train_loss += main_model_loss.item()
        # train_correct += calculate_accuracy(output, target) * len(target)
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------训练生成器--------------------------------------------------------------------------
        # generator_loss = lambda_l1 * gen_loss_l1(masks) + alpha * main_model_loss
        # generator_optimizer.zero_grad()
        # generator_loss.backward()
        # generator_optimizer.step()

        generator_loss = gen_loss_l1(masks)
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------

        train_loss += main_model_loss.item()
        train_correct += calculate_accuracy(output, target) * len(target)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / len(train_loader.dataset)

    # # 自适应调整 L1 损失权重
    # if avg_train_loss > prev_model_loss:
    #     patience = patience - 1
    #     if patience == 0:
    #         print("下降了")
    #         lambda_l1 = lambda_l1 * 0.5  # 减小 L1 正则化权重
    #         # alpha = alpha * 2
    #         patience = 2  # 重置耐心值
    # else:
    #     patience = 2  # 重置耐心值，如果模型性能没有下降

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
        f"Epoch [{epoch + 1}/{num_epochs}], generator Loss: {generator_loss.item():.4f}, model Loss: {main_model_loss.item():.4f}, model Accuracy: {avg_train_acc:.4f}, generator learning rate:{current_lr:.20f}")

# --------------------------------------------------------------------------------------训练结束-----------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------测试阶段-----------------------------------------------------------------------------------


from sklearn.metrics import classification_report

generator.eval()
main_model.eval()
# 使用训练好的生成器生成掩码
z = torch.ones(1, z_dim).to(device)  # 生成随机噪声
# z = torch.ones(1, z_dim)  # 生成随机噪声
masks = generator(z)
print(masks)

original_model = deepcopy(main_model)

# 应用掩码
with torch.no_grad():
    main_model.depthwise1.weight *= masks[0]
    main_model.pointwise1.weight *= masks[1]
    main_model.depthwise2.weight *= masks[2]
    main_model.pointwise2.weight *= masks[3]
    main_model.depthwise3.weight *= masks[4]
    main_model.pointwise3.weight *= masks[5]
    main_model.fc1.weight *= masks[6]
    main_model.fc2.weight *= masks[7]
    main_model.fc3.weight *= masks[8]
    main_model.fc4.weight *= masks[9]

pruned_model = deepcopy(main_model)

# nonzero_params_before = sum(p.numel() for p in original_model.parameters() if p.data.nonzero().numel() > 0)
# print(f"剪枝前非零参数数量: {nonzero_params_before}")
# nonzero_params_after = sum(p.numel() for p in pruned_model.parameters() if p.data.nonzero().numel() > 0)
# print(f"剪枝后非零参数数量: {nonzero_params_after}")




# --------------------------------------------------------------------------------------------------------------------------------
# 打印模型的非零参数数量（剪枝后的有效参数）
def count_nonzero_parameters(model):
    total_nonzero_params = sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)
    return total_nonzero_params


# 计算模型推理时间
def measure_inference_time(model, input_size, device='cpu', iterations=100):
    model.eval()
    x = torch.randn(input_size).to(device)

    # 测量时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    return avg_time


# 打印模型参数数量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


# 测试模型的剪枝效果
def test_pruned_model(original_model, pruned_model, input_size, device='cpu'):
    original_model.to(device)
    pruned_model.to(device)

    # 统计参数数量
    print(f"原始模型参数数量: {count_parameters(original_model)}")
    print(f"剪枝后模型参数数量: {count_parameters(pruned_model)}")

    # 统计非零参数数量
    print(f"原始模型非零参数数量: {count_nonzero_parameters(original_model)}")
    print(f"剪枝后模型非零参数数量: {count_nonzero_parameters(pruned_model)}")

    # 测量推理时间
    original_time = measure_inference_time(original_model, input_size, device=device)
    pruned_time = measure_inference_time(pruned_model, input_size, device=device)

    print(f"原始模型平均推理时间: {original_time:.6f} 秒")
    print(f"剪枝后模型平均推理时间: {pruned_time:.6f} 秒")


test_pruned_model(original_model, pruned_model, (64, 1, 57), 'cuda')

import torch
import torchprofile

# 假设 main_model 是你的模型
dummy_input = torch.randn(64, 1, 57).to(device)  # 根据你的输入形状进行调整
flops_original = torchprofile.profile_macs(original_model, dummy_input)
flops_pruned = torchprofile.profile_macs(pruned_model, dummy_input)

print(f"原始模型FLOPs: {flops_original}")
print(f"剪枝后模型FLOPs: {flops_pruned}")

# --------------------------------------------------------------------------------------------------------------------------------

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

        # outputs = main_model(data)
        outputs = pruned_model(data)

        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        all_preds.extend(predicted.cpu().numpy())  # 将GPU上的张量移回CPU
        all_labels.extend(target.cpu().numpy())

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
# 生成分类报告
print(classification_report(all_labels, all_preds, target_names=class_names))

# class PrunedModel(nn.Module):
#     def __init__(self, original_model, mask):
#         super(PrunedModel, self).__init__()
#
#         # 根据掩码过滤有效的通道
#         keep_channels = mask[0].squeeze().nonzero(as_tuple=True)[0].tolist()
#         in_channels = original_model.depthwise1.in_channels
#         out_channels = len(keep_channels)
#         self.depthwise1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
#                                     padding=1,
#                                     groups=1, bias=False)
#
#         keep_channels = mask[1].squeeze().nonzero(as_tuple=True)[0].tolist()
#         in_channels = original_model.pointwise1.in_channels
#         out_channels = len(keep_channels)
#         self.pointwise1 = nn.Conv1d(in_channels, out_channels, kernel_size=1,
#                                     stride=1, padding=0, bias=False)
#
#         keep_channels = mask[2].squeeze().nonzero(as_tuple=True)[0].tolist()
#         in_channels = original_model.depthwise2.in_channels
#         out_channels = len(keep_channels)
#         self.depthwise2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
#                                     padding=1,
#                                     groups=1, bias=False)
#
#         keep_channels = mask[3].squeeze().nonzero(as_tuple=True)[0].tolist()
#         in_channels = original_model.pointwise2.in_channels
#         out_channels = len(keep_channels)
#         self.pointwise2 = nn.Conv1d(in_channels, out_channels, kernel_size=1,
#                                     stride=1, padding=0, bias=False)
#
#         keep_channels = mask[4].squeeze().nonzero(as_tuple=True)[0].tolist()
#         in_channels = original_model.depthwise3.in_channels
#         out_channels = len(keep_channels)
#         self.depthwise3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
#                                     padding=1,
#                                     groups=1, bias=False)
#
#         keep_channels = mask[5].squeeze().nonzero(as_tuple=True)[0].tolist()
#         in_channels = original_model.pointwise3.in_channels
#         out_channels = len(keep_channels)
#         self.pointwise3 = nn.Conv1d(in_channels, out_channels, kernel_size=1,
#                                     stride=1, padding=0, bias=False)
#
#         # 处理全连接层
#         keep_channels_fc1 = masks[6].squeeze().nonzero(as_tuple=True)[0].tolist()
#         self.fc1 = nn.Linear(
#             in_features=len(keep_channels_fc1),
#             out_features=original_model.fc1.out_features
#         )
#
#         keep_channels_fc2 = masks[7].squeeze().nonzero(as_tuple=True)[0].tolist()
#         self.fc2 = nn.Linear(
#             in_features=len(keep_channels_fc2),
#             out_features=original_model.fc2.out_features
#         )
#
#         keep_channels_fc3 = masks[8].squeeze().nonzero(as_tuple=True)[0].tolist()
#         self.fc3 = nn.Linear(
#             in_features=len(keep_channels_fc3),
#             out_features=original_model.fc3.out_features
#         )
#
#         keep_channels_fc4 = masks[9].squeeze().nonzero(as_tuple=True)[0].tolist()
#         self.fc4 = nn.Linear(
#             in_features=len(keep_channels_fc4),
#             out_features=original_model.fc4.out_features
#         )
#
#     def forward(self, x):
#         # 通过深度卷积层
#         out = self.depthwise1(x)
#         print(out.shape)
#         # 通过逐点卷积层
#         out = self.pointwise1(out)
#         out = F.relu(out)
#         out = F.max_pool1d(out, 2)
#         print(out.shape)
#
#         # 通过深度卷积层
#         out = self.depthwise2(out)
#         # 通过逐点卷积层
#         out = self.pointwise2(out)
#         out = F.relu(out)
#         out = F.max_pool1d(out, 3)
#
#         # 通过深度卷积层
#         out = self.depthwise3(out)
#         # 通过逐点卷积层
#         out = self.pointwise3(out)
#         out = F.relu(out)
#         out = F.max_pool1d(out, 3)
#         # print("形状为：", out.shape)
#         out = out.reshape(self.batch_size, -1)
#
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = F.relu(self.fc3(out))
#         out = self.fc4(out)
#
#         return out
#
#
# def copy_pruned_weights(original_model, pruned_model, mask):
#     with torch.no_grad():
#         # 获取需要保留的通道索引
#         keep_channels = mask[0].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.depthwise1.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.depthwise1.weight.data[:len(keep_channels)] = pruned_weights
#
#         keep_channels = mask[1].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.pointwise1.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.pointwise1.weight.data[:len(keep_channels)] = pruned_weights
#
#         keep_channels = mask[2].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.depthwise2.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.depthwise2.weight.data[:len(keep_channels)] = pruned_weights
#
#         keep_channels = mask[3].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.pointwise2.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.pointwise2.weight.data[:len(keep_channels)] = pruned_weights
#
#         keep_channels = mask[4].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.depthwise3.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.depthwise3.weight.data[:len(keep_channels)] = pruned_weights
#
#         keep_channels = mask[5].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.pointwise3.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.pointwise3.weight.data[:len(keep_channels)] = pruned_weights
#
#         keep_channels = mask[6].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.fc1.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.fc1.weight.data[:len(keep_channels)] = pruned_weights
#
#         keep_channels = mask[7].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.fc2.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.fc2.weight.data[:len(keep_channels)] = pruned_weights
#
#         keep_channels = mask[8].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.fc3.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.fc3.weight.data[:len(keep_channels)] = pruned_weights
#
#         keep_channels = mask[9].squeeze().nonzero(as_tuple=True)[0].tolist()
#         # 剪切原始模型的权重
#         original_weights = original_model.fc4.weight.data
#         pruned_weights = original_weights[keep_channels]
#         # 赋值到新模型的权重
#         pruned_model.fc4.weight.data[:len(keep_channels)] = pruned_weights


# 复制剪枝后的权重
# copy_pruned_weights(main_model, pruned_model, masks)
