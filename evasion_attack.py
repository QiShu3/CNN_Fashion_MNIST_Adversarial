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

# 启用梯度计算，对抗攻击需要计算输入图像的梯度
torch.set_grad_enabled(True)

# 数据转换：将图像转换为 Tensor
transform = transforms.ToTensor()

# 加载测试集
test_set = datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    download=True,
    transform=transform
)

class Network(nn.Module):
    """
    与训练脚本一致的 CNN 结构，用于加载已训练模型并进行攻击测试。
    对抗攻击必须保证模型结构与训练时完全一致，否则权重矩阵维数无法匹配。
    """
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        """
        前向传播：返回未归一化的类别得分 (Logits)
        """
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 12*4*4)))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

# 实例化模型
model = Network()

# 加载保存好的模型文件
PATH = './model1.pth'
# 注意：使用 torch.load 加载整个模型对象
model = torch.load(PATH, weights_only=False)
model.eval() # 必须切换到评估模式

# 攻击测试时，batch_size 通常设为 1，以便逐张图像计算梯度
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

# 定义不同的扰动强度 epsilon
# epsilon=0 代表原始图像，不施加攻击
epsilons = [0, .05, .1, .15, .2, .25, .3]

def fgsm_attack(image, epsilon, data_grad):
    """
    FGSM (Fast Gradient Sign Method) 攻击核心逻辑
    
    原理：利用损失函数相对于输入图像的梯度方向，对图像施加扰动。
    数学公式：x_adv = x + epsilon * sign(grad(J(theta, x, y)))
    
    :param image: 原始图像张量
    :param epsilon: 扰动强度（步长）
    :param data_grad: 损失函数对输入图像求导得到的梯度矩阵
    :return: 施加扰动并裁剪后的对抗样本图像
    """
    # 获取梯度的符号 (1, -1 或 0)
    sign_data_grad = data_grad.sign()
    
    # 沿着梯度上升方向（增加损失的方向）移动像素值
    perturbed_image = image + epsilon * sign_data_grad
    
    # 确保像素值保持在 [0, 1] 合法范围内（图像像素截断）
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def test(model, test_loader, epsilon):
    """
    在给定的 epsilon 强度下，对测试集执行 FGSM 攻击并统计模型表现
    
    :param model: 预训练好的模型
    :param test_loader: 测试数据加载器 (batch_size=1)
    :param epsilon: 扰动强度
    :return: (攻击后的准确率, 部分对抗样本示例)
    """
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        # 核心步骤：设置输入数据需要梯度，否则无法对图像求导
        data.requires_grad = True

        # 1. 原始预测
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        # 如果模型本来就预测错了，则跳过，不计入攻击成功统计
        if init_pred.item() != target.item():
            continue

        # 2. 计算梯度
        # 使用交叉熵损失函数
        loss = F.cross_entropy(output, target)
        model.zero_grad() # 清空模型梯度
        loss.backward()   # 反向传播，计算图像 data 的梯度
        
        # 提取图像的梯度矩阵
        data_grad = data.grad.data

        # 3. 实施攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 4. 对攻击后的样本重新预测
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        # 检查攻击是否成功（预测结果是否仍与真实标签一致）
        if final_pred.item() == target.item():
            correct += 1
            # 记录少量样本用于后续展示
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # 攻击成功的情况
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # 计算最终准确率
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc:.4f}")
    
    return final_acc, adv_examples

# 运行不同强度的攻击测试
accuracies = []
examples = []

for eps in epsilons:
    acc, ex = test(model, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# 可视化：绘制准确率随 epsilon 增加而下降的曲线
plt.figure(figsize=(8, 6))
plt.plot(epsilons, accuracies, "*-", color='blue')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon (FGSM Attack on Fashion-MNIST)")
plt.xlabel("Epsilon (Perturbation Strength)")
plt.ylabel("Model Accuracy")
plt.grid(True, linestyle='--', alpha=0.6)

# 将结果保存到文件
save_path = "saved/accuracy_epsilon.png"
plt.savefig(save_path, dpi=300)
print(f"\nResult plot saved to: {save_path}")

# 历史运行结果参考：
"""
Epsilon: 0      Test Accuracy = 8922 / 10000 = 0.8922 (原始表现)
Epsilon: 0.05   Test Accuracy = 3595 / 10000 = 0.3595 (大幅下降)
Epsilon: 0.1    Test Accuracy = 1663 / 10000 = 0.1663
Epsilon: 0.15   Test Accuracy = 784 / 10000 = 0.0784
Epsilon: 0.2    Test Accuracy = 361 / 10000 = 0.0361
Epsilon: 0.25   Test Accuracy = 230 / 10000 = 0.0230
Epsilon: 0.3    Test Accuracy = 181 / 10000 = 0.0181 (几乎完全瘫痪)
"""
