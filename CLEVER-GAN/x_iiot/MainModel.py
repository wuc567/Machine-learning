import torch
import torch.nn as nn
import torch.nn.functional as F


# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, batch_size):
        super(DepthwiseSeparableConv, self).__init__()
        self.batch_size=batch_size
        # 深度卷积：对每个输入通道独立卷积
        self.depthwise1 = nn.Conv1d(1, 1, kernel_size=3,
                                    stride=1, padding=1, groups=1, bias=False)
        # 逐点卷积：用 1x1 卷积核将通道维度进行线性组合
        self.pointwise1 = nn.Conv1d(1, 16, kernel_size=1,
                                    stride=1, padding=0, bias=False)
        # 深度卷积：对每个输入通道独立卷积
        self.depthwise2 = nn.Conv1d(16, 16, kernel_size=3,
                                    stride=1, padding=1, groups=16, bias=False)
        # 逐点卷积：用 1x1 卷积核将通道维度进行线性组合
        self.pointwise2 = nn.Conv1d(16, 32, kernel_size=1,
                                    stride=1, padding=0, bias=False)
        # 深度卷积：对每个输入通道独立卷积
        self.depthwise3 = nn.Conv1d(32, 32, kernel_size=3,
                                    stride=1, padding=1, groups=32, bias=False)
        # 逐点卷积：用 1x1 卷积核将通道维度进行线性组合
        self.pointwise3 = nn.Conv1d(32, 64, kernel_size=1,
                                    stride=1, padding=0, bias=False)

        self.fc1 = nn.Linear(192, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 10)

    def forward(self, x):
        # 通过深度卷积层
        out = self.depthwise1(x)
        # 通过逐点卷积层
        out = self.pointwise1(out)
        out = F.relu(out)
        out = F.max_pool1d(out, 2)

        # 通过深度卷积层
        out = self.depthwise2(out)
        # 通过逐点卷积层
        out = self.pointwise2(out)
        out = F.relu(out)
        out = F.max_pool1d(out, 3)

        # 通过深度卷积层
        out = self.depthwise3(out)
        # 通过逐点卷积层
        out = self.pointwise3(out)
        out = F.relu(out)
        out = F.max_pool1d(out, 3)
        # print("形状为：", out.shape)
        out = out.reshape(self.batch_size, -1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out


# 测试代码
if __name__ == '__main__':
    # 假设输入数据为 (batch_size, channels, height, width) = (1, 32, 112, 112)
    x = torch.randn(1, 1, 57)

    # 创建深度可分离卷积实例，输入通道数为 32，输出通道数为 64
    model = DepthwiseSeparableConv(1)

    # 前向传播
    output = model(x)

    # 输出的特征图大小
    print(output.shape)  # 预期输出: torch.Size([1, 64, 112, 112])
