import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from itertools import product

# 启用全局梯度计算（在训练脚本中默认开启）
torch.set_grad_enabled(True)

class Network(nn.Module):
    """
    卷积神经网络 (CNN)：用于 Fashion-MNIST 图像分类
    
    结构设计：
    1. 卷积层 1: 1个输入通道 (灰度图), 6个输出通道, 5x5 卷积核
    2. 卷积层 2: 6个输入通道, 12个输出通道, 5x5 卷积核
    3. 全连接层 1: 输入 12*4*4 (展平后的特征), 输出 120
    4. 全连接层 2: 输入 120, 输出 60
    5. 输出层: 输入 60, 输出 10 (对应 10 个类别)
    """
    def __init__(self):
        super(Network, self).__init__()
        # 定义卷积层：提取空间特征
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        # 定义全连接层：进行分类决策
        # 注意：这里的 12*4*4 是根据输入图像大小 (28x28) 经过两层卷积和池化后推导出来的
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        """
        定义前向传播逻辑
        :param t: 输入张量 [batch_size, 1, 28, 28]
        :return: 未归一化的预测分数 (Logits)
        """
        # 第一层卷积 + ReLU 激活 + 最大池化 (2x2, 步长 2)
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 第二层卷积 + ReLU 激活 + 最大池化 (2x2, 步长 2)
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 展平特征向量，为进入全连接层做准备
        t = F.relu(self.fc1(t.reshape(-1, 12*4*4)))
        # 隐藏层全连接
        t = F.relu(self.fc2(t))
        # 输出层（不加激活函数，因为后面会用 CrossEntropyLoss）
        t = self.out(t)
        return t

def get_num_correct(preds, labels):
    """
    计算预测正确的样本数量
    :param preds: 模型的原始输出张量
    :param labels: 真实标签张量
    :return: 预测正确的整数数量
    """
    return preds.argmax(dim=1).eq(labels).sum().item()

# 1. 数据准备
# 将 PIL 图像转换为张量并归一化到 [0, 1] 范围
transform = transforms.ToTensor()

# 下载并加载 Fashion-MNIST 训练集
train_set = datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transform
)

# 下载并加载 Fashion-MNIST 测试集
test_set = datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    download=True,
    transform=transform
)

# 定义数据采样器：将训练集划分为训练部分 (48000) 和验证部分 (12000)
train_sampler = SubsetRandomSampler(list(range(48000)))
valid_sampler = SubsetRandomSampler(list(range(48000, 60000))) # 修正：验证集索引应接在训练集之后

# 2. 超参数配置
parameters = dict(
    lr = [0.001],      # 学习率
    batch_size = [20]  # 批处理大小
)

param_values = [v for v in parameters.values()]
model = Network()

# 3. 训练循环
# 使用 product 处理多组超参数组合（当前仅一组）
for lr, batch_size in product(*param_values):
    # 初始化数据加载器
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler)
    
    # 初始化 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(10): # 训练 10 轮
        total_loss, total_correct = 0, 0
        valid_loss, valid_correct = 0, 0

        # --- 训练阶段 ---
        model.train() # 切换到训练模式
        for batch in train_loader:
            images, labels = batch

            # 前向传播
            preds = model(images)
            loss = F.cross_entropy(preds, labels) # 计算交叉熵损失

            # 反向传播与优化
            optimizer.zero_grad()  # 清空旧梯度
            loss.backward()        # 计算当前梯度
            optimizer.step()       # 更新权重参数

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
        
        # --- 验证阶段 ---
        model.eval() # 切换到评估模式（关闭 Dropout 等）
        with torch.no_grad(): # 验证时不计算梯度，节省内存和时间
            for batch in valid_loader:
                images, labels = batch

                preds = model(images)
                loss = F.cross_entropy(preds, labels)

                valid_loss += loss.item()
                valid_correct += get_num_correct(preds, labels)

        # 打印当前轮次的统计信息
        print(f"Epoch: {epoch} | "
              f"Train Correct: {total_correct} | Train Loss: {total_loss:.4f} | "
              f"Valid Correct: {valid_correct} | Valid Loss: {valid_loss:.4f}")

    print(f'Training Finished. Lr: {lr}, Batch Size: {batch_size}')

# 4. 测试与评估
all_preds = []
targets = []
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

model.eval()
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch

        preds = model(images)
        # 获取预测类别索引
        all_preds.append(preds.argmax(dim=1))
        targets.append(labels)

# 合并所有批次的预测结果
all_preds = torch.cat(all_preds)
targets = torch.cat(targets)

# 计算混淆矩阵和准确率
cm = confusion_matrix(targets.cpu(), all_preds.cpu())
accuracy = accuracy_score(targets.cpu(), all_preds.cpu())

print("\nConfusion Matrix: \n", cm)
print(f"Overall Accuracy on Test Set: {accuracy:.4%}")

# 5. 保存模型
# 将训练好的模型权重和结构保存到根目录
PATH = './model1.pth'
torch.save(model, PATH)
print(f"Model saved to {PATH}")
