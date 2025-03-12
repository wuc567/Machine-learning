import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=9, kernel_size=20, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=24, kernel_size=20, stride=1, padding=1)

        # self.fc1 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(24, 10)
        # self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # # 应用第一层卷积，并使用ReLU激活函数
        # x = F.relu(self.conv1(x))
        # # 使用2x2的池化层进行下采样
        # x = F.max_pool1d(x, 2)
        # # 应用第二层卷积，并使用ReLU激活函数
        # x = F.relu(self.conv2(x))
        # # 再次使用2x2的池化层进行下采样
        # x = F.max_pool1d(x, 2)
        # # 展平多维的卷积层输出为一维
        #
        # x = F.relu(self.conv3(x))
        # # 再次使用2x2的池化层进行下采样
        # x = F.max_pool1d(x, 2)
        # # 展平多维的卷积层输出为一维
        #
        # x = F.relu(self.conv4(x))
        # # 再次使用2x2的池化层进行下采样
        # x = F.max_pool1d(x, 2)
        # # 展平多维的卷积层输出为一维
        #
        # x = F.relu(self.conv5(x))
        # # 再次使用2x2的池化层进行下采样
        # x = F.max_pool1d(x, 2)
        # # 展平多维的卷积层输出为一维

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)



        x = x.view(16, -1)

        x = self.fc1(x)

        return x
