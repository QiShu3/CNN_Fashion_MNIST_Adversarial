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
#from resources.plotcm import plot_confusion_matrix
from itertools import product
torch.set_grad_enabled(True)

transform = transforms.ToTensor()

test_set = datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=False
    ,download=True
    ,transform = transform)
train_sampler = SubsetRandomSampler(list(range(48000)))
valid_sampler = SubsetRandomSampler(list(range(12000)))

class Network(nn.Module):
    """与训练脚本一致的CNN，用于加载已训练模型并进行攻击测试"""
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        """前向传播：返回未归一化的类别logits"""
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 12*4*4)))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

model = Network()
#Load model from repository root
PATH = './model1.pth'
model = torch.load(PATH, weights_only=False)
model.eval()


test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

''' 
all_preds = []
targets = []
for batch in test_loader:
    images, labels = batch


    preds = model(images)
    loss = F.cross_entropy(preds, labels)

    all_preds.append(torch.max(preds, dim=1).indices)
    targets.append(labels.data)
all_preds = torch.cat(all_preds)
targets = torch.cat(targets)
cm = confusion_matrix(targets, all_preds)
accuracy = accuracy_score(targets, all_preds) 

print("Confusion_matrix: \n", cm)
print("Overall accuracy on test set: ", accuracy)'''


epsilons = [0, .05, .1, .15, .2, .25, .3]
adv_examples = []

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    """FGSM攻击：按梯度符号扰动图像并裁剪到[0,1]"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def test(model, test_loader, epsilon):
    """在给定epsilon下对测试集执行FGSM并统计准确率"""
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        # 与训练一致使用交叉熵以避免log_softmax不匹配
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
# 非交互环境下直接保存图像
plt.savefig("saved/accuracy_epsilon.png", dpi=300)

#after running adversarial attack here are the results
'''
Epsilon: 0      Test Accuracy = 8922 / 10000 = 0.8922
Epsilon: 0.05   Test Accuracy = 3595 / 10000 = 0.3595
Epsilon: 0.1    Test Accuracy = 1663 / 10000 = 0.1663
Epsilon: 0.15   Test Accuracy = 784 / 10000 = 0.0784
Epsilon: 0.2    Test Accuracy = 361 / 10000 = 0.0361
Epsilon: 0.25   Test Accuracy = 230 / 10000 = 0.023
Epsilon: 0.3    Test Accuracy = 181 / 10000 = 0.0181
'''
