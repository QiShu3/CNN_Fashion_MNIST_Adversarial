"""
训练4个不同的CNN模型在Fashion-MNIST数据集上
输出指标：准确度、损失、混淆矩阵
保存模型为.pt文件
使用阶梯式学习率衰减（每10个epoch × 0.1）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

# 尝试导入可选依赖
try:
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("警告: sklearn未安装，将使用基础实现")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，将跳过绘图")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("警告: seaborn未安装，将使用matplotlib绘图")

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from mnist_reader import load_mnist
from models import SimpleCNN, CNNWithBatchNorm, DeepCNN, ResNet

# 如果没有sklearn，实现基础版本
if not HAS_SKLEARN:
    def confusion_matrix(y_true, y_pred):
        """基础混淆矩阵实现"""
        num_classes = len(np.unique(y_true))
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        return cm
    
    def accuracy_score(y_true, y_pred):
        """基础准确度计算"""
        return np.mean(np.array(y_true) == np.array(y_pred))
    
    def classification_report(y_true, y_pred, target_names=None):
        """基础分类报告"""
        cm = confusion_matrix(y_true, y_pred)
        num_classes = cm.shape[0]
        report = []
        report.append("分类报告:\n")
        if target_names is None:
            target_names = [f"类别 {i}" for i in range(num_classes)]
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            report.append(f"{target_names[i]}: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}\n")
        return "".join(report)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 数据集配置
BATCH_SIZE = 64
NUM_EPOCHS = 10  # 10个epoch
LEARNING_RATE = 0.001
LR_SCHEDULE_TYPE = 'constant'  # 固定学习率，不衰减
WEIGHT_DECAY = 1e-4  # L2正则化

# 类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_fashion_mnist(data_path='./data/fashion'):
    """从本地路径加载Fashion-MNIST数据集"""
    print(f'从 {data_path} 加载数据集...')
    
    # 使用mnist_reader加载数据
    X_train, y_train = load_mnist(data_path, kind='train')
    X_test, y_test = load_mnist(data_path, kind='t10k')
    
    # 将数据reshape为28x28并归一化到[0, 1]，然后转换为[-1, 1]
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    # 归一化到[-1, 1]
    X_train = (X_train - 0.5) / 0.5
    X_test = (X_test - 0.5) / 0.5
    
    # 转换为PyTorch张量
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 添加weight_decay进行L2正则化
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器：固定学习率（不使用scheduler）
    scheduler = None
    
    # 记录训练历史
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    learning_rates = []
    
    print(f'\n开始训练 {model_name}...')
    print('=' * 60)
    print(f'初始学习率: {LEARNING_RATE} (固定不变)')
    print(f'权重衰减 (weight_decay): {WEIGHT_DECAY}')
    print(f'学习率调度: 固定学习率，10个epoch内保持{LEARNING_RATE}不变')
    print('=' * 60)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # 计算训练指标
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # 验证阶段
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        # 计算测试指标
        epoch_test_loss = test_loss / len(test_loader)
        epoch_test_acc = 100 * correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)
        
        # 更新学习率（固定学习率，不需要更新）
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        if scheduler is not None:
            scheduler.step()
        
        # 每个epoch都打印（因为只有10个epoch）
        print(f'Epoch [{epoch+1}/{num_epochs}] - 学习率: {current_lr:.6f}')
        print(f'  训练损失: {epoch_train_loss:.4f}, 训练准确度: {epoch_train_acc:.2f}%')
        print(f'  测试损失: {epoch_test_loss:.4f}, 测试准确度: {epoch_test_acc:.2f}%')
    
    print('=' * 60)
    print(f'{model_name} 训练完成!')
    print(f'最终测试准确度: {test_accuracies[-1]:.2f}%')
    print(f'最终测试损失: {test_losses[-1]:.4f}')
    
    return {
        'model': model,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'learning_rates': learning_rates
    }


def evaluate_model(model, test_loader):
    """评估模型并生成混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算准确度
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, cm, all_labels, all_preds


def plot_confusion_matrix(cm, model_name, class_names=class_names):
    """绘制混淆矩阵"""
    if not HAS_MATPLOTLIB:
        print(f'跳过绘制混淆矩阵（matplotlib未安装）')
        return
    
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.yticks(range(len(class_names)), class_names)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.title(f'{model_name} - 混淆矩阵', fontsize=16, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'混淆矩阵已保存到: results/{model_name}_confusion_matrix.png')


def save_results(model_name, results, accuracy, cm, labels, preds):
    """保存结果到文件"""
    os.makedirs('results', exist_ok=True)
    
    # 保存模型
    model_path = f'results/{model_name}.pt'
    torch.save(results['model'].state_dict(), model_path)
    print(f'模型已保存到: {model_path}')
    
    # 保存指标到文本文件
    with open(f'results/{model_name}_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f'{model_name} 评估结果\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'最终测试准确度: {accuracy:.4f} ({accuracy*100:.2f}%)\n')
        f.write(f'最终测试损失: {results["test_losses"][-1]:.4f}\n\n')
        
        f.write('训练历史:\n')
        f.write('Epoch\t训练损失\t训练准确度\t测试损失\t测试准确度\n')
        for i in range(len(results['train_losses'])):
            f.write(f'{i+1}\t{results["train_losses"][i]:.4f}\t'
                   f'{results["train_accuracies"][i]:.2f}%\t'
                   f'{results["test_losses"][i]:.4f}\t'
                   f'{results["test_accuracies"][i]:.2f}%\n')
        
        f.write('\n混淆矩阵:\n')
        f.write('\t' + '\t'.join(class_names) + '\n')
        for i, row in enumerate(cm):
            f.write(f'{class_names[i]}\t' + '\t'.join(map(str, row)) + '\n')
        
        # 分类报告
        f.write('\n分类报告:\n')
        f.write(classification_report(labels, preds, target_names=class_names))
    
    print(f'指标已保存到: results/{model_name}_metrics.txt')
    
    # 生成YAML配置文件
    generate_yaml_config(model_name, results, accuracy, cm)


def generate_yaml_config(model_name, results, accuracy, cm):
    """为每个模型生成YAML配置文件"""
    try:
        import yaml
        has_yaml = True
    except ImportError:
        has_yaml = False
        import json
    
    # 计算每个类别的指标
    num_classes = cm.shape[0]
    class_metrics = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics.append({
            'class_id': int(i),
            'class_name': class_names[i],
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        })
    
    # 构建YAML配置
    config = {
        'model': {
            'name': model_name,
            'model_file': f'{model_name}.pt',
            'num_classes': 10,
            'input_size': [1, 28, 28]
        },
        'training': {
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'initial_learning_rate': LEARNING_RATE,
            'learning_rate_schedule': {
                'type': LR_SCHEDULE_TYPE,
                'description': '10个epoch内学习率保持0.001不变，避免lr降为0'
            },
            'weight_decay': WEIGHT_DECAY,
            'dropout_rate': 0.15
        },
        'results': {
            'final_test_accuracy': float(accuracy),
            'final_test_accuracy_percent': float(accuracy * 100),
            'final_test_loss': float(results['test_losses'][-1]),
            'final_train_accuracy': float(results['train_accuracies'][-1]),
            'final_train_loss': float(results['train_losses'][-1]),
            'best_test_accuracy': float(max(results['test_accuracies'])),
            'best_test_accuracy_epoch': int(np.argmax(results['test_accuracies']) + 1),
            'best_test_loss': float(min(results['test_losses'])),
            'best_test_loss_epoch': int(np.argmin(results['test_losses']) + 1)
        },
        'class_metrics': class_metrics,
        'training_history': {
            'epochs': list(range(1, len(results['train_losses']) + 1)),
            'train_losses': [float(x) for x in results['train_losses']],
            'train_accuracies': [float(x) for x in results['train_accuracies']],
            'test_losses': [float(x) for x in results['test_losses']],
            'test_accuracies': [float(x) for x in results['test_accuracies']],
            'learning_rates': [float(x) for x in results.get('learning_rates', [])]
        },
        'confusion_matrix': {
            'matrix': cm.tolist(),
            'class_names': class_names
        }
    }
    
    # 保存配置文件
    if has_yaml:
        yaml_path = f'results/{model_name}_config.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f'YAML配置文件已保存到: {yaml_path}')
    else:
        # 如果没有yaml库，使用JSON格式作为替代
        json_path = f'results/{model_name}_config.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f'JSON配置文件已保存到: {json_path} (yaml库未安装，使用JSON格式)')


def plot_training_history(results_dict):
    """绘制所有模型的训练历史对比"""
    if not HAS_MATPLOTLIB:
        print('跳过绘制训练历史对比图（matplotlib未安装）')
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失
    ax = axes[0, 0]
    for name, results in results_dict.items():
        ax.plot(results['train_losses'], label=name)
    ax.set_title('训练损失', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # 测试损失
    ax = axes[0, 1]
    for name, results in results_dict.items():
        ax.plot(results['test_losses'], label=name)
    ax.set_title('测试损失', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # 训练准确度
    ax = axes[1, 0]
    for name, results in results_dict.items():
        ax.plot(results['train_accuracies'], label=name)
    ax.set_title('训练准确度', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True)
    
    # 测试准确度
    ax = axes[1, 1]
    for name, results in results_dict.items():
        ax.plot(results['test_accuracies'], label=name)
    ax.set_title('测试准确度', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('训练历史对比图已保存到: results/training_comparison.png')


def main():
    """主函数"""
    print('=' * 60)
    print('Fashion-MNIST CNN模型训练')
    print('=' * 60)
    
    # 加载数据
    print('\n加载数据集...')
    data_path = './data/fashion'
    train_loader, test_loader = load_fashion_mnist(data_path)
    print(f'训练集大小: {len(train_loader.dataset)}')
    print(f'测试集大小: {len(test_loader.dataset)}')
    
    # 定义模型
    models = {
        'SimpleCNN': SimpleCNN(num_classes=10),
        'CNNWithBatchNorm': CNNWithBatchNorm(num_classes=10),
        'DeepCNN': DeepCNN(num_classes=10),
        'ResNet': ResNet(num_classes=10, num_blocks=[2, 2, 2, 2])
    }
    
    # 训练所有模型
    all_results = {}
    for model_name, model in models.items():
        # 训练模型
        results = train_model(model, train_loader, test_loader, model_name)
        
        # 评估模型
        accuracy, cm, labels, preds = evaluate_model(results['model'], test_loader)
        print(f'\n{model_name} 最终测试准确度: {accuracy*100:.2f}%')
        
        # 绘制混淆矩阵
        plot_confusion_matrix(cm, model_name)
        
        # 保存结果
        save_results(model_name, results, accuracy, cm, labels, preds)
        
        all_results[model_name] = results
        print()
    
    # 绘制训练历史对比
    plot_training_history(all_results)
    
    # 打印总结
    print('\n' + '=' * 60)
    print('训练完成总结')
    print('=' * 60)
    for model_name, results in all_results.items():
        print(f'{model_name}:')
        print(f'  最终测试准确度: {results["test_accuracies"][-1]:.2f}%')
        print(f'  最终测试损失: {results["test_losses"][-1]:.4f}')
        print(f'  模型文件: results/{model_name}.pt')
        print()
    
    print('所有结果已保存到 results/ 目录')


if __name__ == '__main__':
    main()

