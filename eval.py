import os
import csv
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'fashion-mnist-master'))
from models import SimpleCNN, ResNet, CNNWithBatchNorm, DeepCNN

from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image

FASHION_MNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
LABEL_NAME_TO_ID = {name: idx for idx, name in enumerate(FASHION_MNIST_LABELS)}
LABEL_NAME_TO_ID["T-shirt"] = LABEL_NAME_TO_ID["T-shirt/top"]  # Handle folder name mismatch


def load_model(model_path: str, model_name: str, device: torch.device) -> nn.Module:
    """加载训练好的模型；根据模型名称初始化对应结构"""
    # 实例化模型
    if model_name == "ResNet":
        model = ResNet()
    elif model_name == "SimpleCNN":
        model = SimpleCNN()
    elif model_name == "CNNWithBatchNorm":
        model = CNNWithBatchNorm()
    elif model_name == "DeepCNN":
        model = DeepCNN()
    else:
        # 默认回退逻辑，或者抛错
        print(f"Unknown model name {model_name}, trying to auto-detect not supported well. Defaulting to SimpleCNN.")
        model = SimpleCNN()

    model = model.to(device)
    
    # 加载权重
    try:
        # 尝试直接加载整模型（如果保存的是整模型）
        loaded_obj = torch.load(model_path, map_location=device)
        if isinstance(loaded_obj, dict):
            # 是 state_dict
            state = loaded_obj
        elif isinstance(loaded_obj, nn.Module):
            # 是整模型，直接返回（这种情况下上面的实例化可能浪费了，但兼容性更好）
            print(f"Loaded full model object for {model_name}")
            loaded_obj.eval()
            return loaded_obj
        else:
            raise ValueError(f"Unknown loaded object type: {type(loaded_obj)}")
            
        # 加载 state_dict
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            print(f"Strict loading failed: {e}. Trying strict=False...")
            model.load_state_dict(state, strict=False)
            
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise e

    model.eval()
    return model

def load_images(root: str) -> List[Tuple[str, torch.Tensor, Optional[int], Optional[str]]]:
    """递归加载PNG图片为张量；返回 (路径, 张量, 标签id或None, 标签名或None)"""
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # [0,1], shape [1,28,28]
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1] matching training
    ])
    results: List[Tuple[str, torch.Tensor, Optional[int], Optional[str]]] = []
    print(f"Scanning images in {root}...")
    for dirpath, _, filenames in os.walk(root):
        # 通过上级目录名识别类别
        parts = dirpath.replace("\\", "/").split("/")
        label_name = None
        label_id = None
        for p in parts[::-1]:
            if p in LABEL_NAME_TO_ID:
                label_name = p
                label_id = LABEL_NAME_TO_ID[p]
                break
        for fn in filenames:
            if not fn.lower().endswith(".png"):
                continue
            fp = os.path.join(dirpath, fn)
            img = Image.open(fp).convert("L")
            tensor = tfm(img).unsqueeze(0)  # [1,1,28,28]
            results.append((fp, tensor, label_id, label_name))
    print(f"Found {len(results)} images.")
    return results

def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """FGSM攻击：按梯度符号扰动图像，并使用掩膜过滤背景噪点
    
    返回:
    - perturbed_image: 对抗样本
    - mask: 使用的背景掩膜
    """
    sign_data_grad = data_grad.sign()
    
    # 生成掩膜：仅保留非背景像素（假设背景为纯黑0）
    # 使用一个小阈值（如0.01）避免浮点误差
    mask = (image > 0.01).float()
    
    # 应用掩膜：仅在前景区域施加扰动
    masked_perturbation = epsilon * sign_data_grad * mask
    
    perturbed_image = image + masked_perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, mask

def fgsm_targeted_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """靶向FGSM攻击：沿着梯度反方向扰动图像以最小化目标类别的Loss
    
    返回:
    - perturbed_image: 对抗样本
    - mask: 使用的背景掩膜
    """
    sign_data_grad = data_grad.sign()
    
    # 生成掩膜：仅保留非背景像素（假设背景为纯黑0）
    mask = (image > 0.01).float()
    
    # 应用掩膜：仅在前景区域施加扰动
    # 靶向攻击：朝着梯度反方向移动 (Minimize Loss of Target Class) -> x - epsilon * sign(grad)
    masked_perturbation = epsilon * sign_data_grad * mask
    
    perturbed_image = image - masked_perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, mask

def iterative_targeted_fgsm_attack(
    image: torch.Tensor,
    epsilon: float,
    iters: int,
    alpha: Optional[float],
    target_label: int,
    model: nn.Module,
    device: torch.device,
    use_mask: bool = True,
    random_start: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """靶向I-FGSM/PGD迭代式攻击：使预测结果逼近 target_label"""
    x0 = image.to(device)
    mask = (x0 > 0.01).float() if use_mask else torch.ones_like(x0)
    x_adv = x0.clone().detach()
    
    if random_start:
        rand = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x0 + rand, 0, 1)
        
    if alpha is None:
        alpha = epsilon / max(iters, 1)
        
    target = torch.tensor([target_label], dtype=torch.long, device=device)
    
    for _ in range(iters):
        x_adv = x_adv.detach()
        x_adv.requires_grad = True
        logits = model(x_adv)
        
        # 靶向：最小化目标类别的 Loss
        loss = F.cross_entropy(logits, target)
        
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        step = grad.sign()
        
        if use_mask:
            step = step * mask
            
        # 靶向攻击：梯度下降 (x - alpha * sign(grad))
        x_adv = x_adv.detach() - alpha * step
        
        # 投影回 L∞ 球并裁剪到[0,1]
        delta = torch.clamp(x_adv - x0, -epsilon, epsilon)
        x_adv = torch.clamp(x0 + delta, 0, 1)
        
    return x_adv.detach(), mask

def iterative_fgsm_attack(
    image: torch.Tensor,
    epsilon: float,
    iters: int,
    alpha: Optional[float],
    model: nn.Module,
    device: torch.device,
    use_mask: bool = True,
    random_start: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """I-FGSM/PGD迭代式攻击：在 L∞ 约束内多步更新并裁剪到[0,1]
    
    返回:
    - adv_image: 迭代攻击生成的对抗样本
    - mask: 使用的背景掩膜
    """
    x0 = image.to(device)
    mask = (x0 > 0.01).float() if use_mask else torch.ones_like(x0)
    x_adv = x0.clone().detach()
    if random_start:
        # 在 L∞ 球内随机初始化
        rand = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x0 + rand, 0, 1)
    if alpha is None:
        alpha = epsilon / max(iters, 1)
    for _ in range(iters):
        x_adv = x_adv.detach()
        x_adv.requires_grad = True
        logits = model(x_adv)
        # 非定向（untargeted）：最大化当前预测的损失
        target = logits.argmax(dim=1)
        loss = F.cross_entropy(logits, target)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        step = grad.sign()
        if use_mask:
            step = step * mask
        x_adv = x_adv.detach() + alpha * step
        # 投影回 L∞ 球并裁剪到[0,1]
        delta = torch.clamp(x_adv - x0, -epsilon, epsilon)
        x_adv = torch.clamp(x0 + delta, 0, 1)
    return x_adv.detach(), mask

def attack_one(model: nn.Module, image: torch.Tensor, label_id: Optional[int], epsilon: float, device: torch.device) -> Dict:
    """对单张图片执行攻击并返回前后预测、是否成功与对抗样本"""
    image = image.to(device)
    image.requires_grad = True

    logits = model(image)
    pred_before = logits.argmax(dim=1).item()

    if label_id is None:
        target = torch.tensor([pred_before], dtype=torch.long, device=device)
    else:
        target = torch.tensor([label_id], dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, target)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data

    adv_image, mask = fgsm_attack(image, epsilon, data_grad)
    logits_adv = model(adv_image)
    pred_after = logits_adv.argmax(dim=1).item()
    success = (pred_after != pred_before) if label_id is None else (pred_after != label_id)

    return {
        "pred_before": pred_before,
        "pred_after": pred_after,
        "success": success,
        "adv_image": adv_image.detach().cpu(),
        "mask": mask.detach().cpu()
    }

def attack_one_targeted(
    model: nn.Module,
    image: torch.Tensor,
    target_label: int,
    epsilon: float,
    device: torch.device
) -> Dict:
    """对单张图片执行靶向攻击并返回前后预测、是否成功与对抗样本"""
    image = image.to(device)
    image.requires_grad = True

    logits = model(image)
    pred_before = logits.argmax(dim=1).item()

    # 靶向攻击：目标是 target_label
    target = torch.tensor([target_label], dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, target)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data

    # 使用靶向 FGSM
    adv_image, mask = fgsm_targeted_attack(image, epsilon, data_grad)
    
    logits_adv = model(adv_image)
    pred_after = logits_adv.argmax(dim=1).item()
    
    # 靶向攻击成功条件：最终预测类别 == 目标类别
    success = (pred_after == target_label)

    return {
        "pred_before": pred_before,
        "pred_after": pred_after,
        "success": success,
        "adv_image": adv_image.detach().cpu(),
        "mask": mask.detach().cpu()
    }


def attack_one_iter_targeted(
    model: nn.Module,
    image: torch.Tensor,
    target_label: int,
    epsilon: float,
    iters: int,
    alpha: Optional[float],
    device: torch.device,
    use_mask: bool = True,
    random_start: bool = False,
) -> Dict:
    """靶向迭代式攻击封装"""
    image = image.to(device)
    logits = model(image)
    pred_before = logits.argmax(dim=1).item()
    
    adv_image, mask = iterative_targeted_fgsm_attack(
        image=image,
        epsilon=epsilon,
        iters=iters,
        alpha=alpha,
        target_label=target_label,
        model=model,
        device=device,
        use_mask=use_mask,
        random_start=random_start,
    )
    
    logits_adv = model(adv_image)
    pred_after = logits_adv.argmax(dim=1).item()
    
    # 靶向攻击成功条件：最终预测类别 == 目标类别
    success = (pred_after == target_label)

    return {
        "pred_before": pred_before,
        "pred_after": pred_after,
        "success": success,
        "adv_image": adv_image.detach().cpu(),
        "mask": mask.detach().cpu()
    }

def attack_one_iter(
    model: nn.Module,
    image: torch.Tensor,
    label_id: Optional[int],
    epsilon: float,
    iters: int,
    alpha: Optional[float],
    device: torch.device,
    use_mask: bool = True,
    random_start: bool = False,
) -> Dict:
    """迭代式攻击封装：返回前后预测、是否成功与对抗样本"""
    image = image.to(device)
    logits = model(image)
    pred_before = logits.argmax(dim=1).item()
    adv_image, mask = iterative_fgsm_attack(
        image=image,
        epsilon=epsilon,
        iters=iters,
        alpha=alpha,
        model=model,
        device=device,
        use_mask=use_mask,
        random_start=random_start,
    )
    logits_adv = model(adv_image)
    pred_after = logits_adv.argmax(dim=1).item()
    success = (pred_after != pred_before) if label_id is None else (pred_after != label_id)

    return {
        "pred_before": pred_before,
        "pred_after": pred_after,
        "success": success,
        "adv_image": adv_image.detach().cpu(),
        "mask": mask.detach().cpu()
    }


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def cleanup_results(root_dir: str):
    """清理目录下的 png 和 csv 文件，保留子目录结构"""
    if not os.path.exists(root_dir):
        return
    print(f"Cleaning up old results in {root_dir}...")
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith((".png", ".csv")):
                try:
                    os.remove(os.path.join(dirpath, fn))
                except Exception as e:
                    print(f"Failed to delete {fn}: {e}")


def calculate_and_print_metrics(model_name: str, attack_type: str, results: List[Dict], total_l2: float):
    """统计并打印攻击指标：Clean Acc, Adv Acc, ASR, Avg L2"""
    total = len(results)
    if total == 0:
        print(f"[{model_name} | {attack_type}] 未处理任何图片。")
        return

    # 统计各项指标
    clean_correct = sum(1 for r in results if r["clean_correct"])
    adv_correct = sum(1 for r in results if r["adv_correct"])
    
    success_on_clean = sum(1 for r in results if r["clean_correct"] and not r["adv_correct"])
    
    clean_acc = clean_correct / total * 100
    adv_acc = adv_correct / total * 100
    asr = (success_on_clean / clean_correct * 100) if clean_correct > 0 else 0.0
    
    avg_l2 = total_l2 / total

    print("\n" + "="*55)
    print(f"  结果统计: {model_name} | {attack_type}")
    print("="*55)
    print(f"  Clean Accuracy       : {clean_acc:6.2f}% ({clean_correct}/{total})")
    print(f"  Adversarial Accuracy : {adv_acc:6.2f}% ({adv_correct}/{total})")
    print(f"  Attack Success Rate  : {asr:6.2f}% (Base: {clean_correct} correct)")
    print(f"  Avg L2 Perturbation  : {avg_l2:.4f}")
    print("="*55 + "\n")


def main():
    """批量攻击与结果评估（支持多种攻击方法对比）"""
    # 自动扫描 results50epoch 目录下的模型
    base_results_dir = "/Users/qishu/Desktop/一次性/研一上/矩阵理论汇报/CNN_pytorch_adversarial_attack_Fashion_MNIST/fashion-mnist-master/results50epoch"
    models_to_evaluate = []
    
    if os.path.exists(base_results_dir):
        for model_name in os.listdir(base_results_dir):
            model_dir = os.path.join(base_results_dir, model_name)
            if os.path.isdir(model_dir):
                pt_file = os.path.join(model_dir, f"{model_name}.pt")
                if os.path.exists(pt_file):
                    models_to_evaluate.append({"name": model_name, "path": pt_file})
    
    if not models_to_evaluate:
        print(f"Warning: No models found in {base_results_dir}. Using default list.")
        models_to_evaluate = [
            {"name": "SimpleCNN", "path": "fashion-mnist-master/results50epoch/SimpleCNN/SimpleCNN.pt"},
            {"name": "ResNet", "path": "fashion-mnist-master/results50epoch/ResNet/ResNet.pt"}
        ]

    # 攻击配置
    epsilon = 0.15
    attack_methods = [
        {"type": "FGSM", "params": {}},
        {"type": "I-FGSM", "params": {"iters": 10, "alpha": 0.01}}
    ]
    
    out_dir = "saved/adversarial"
    ensure_dir(out_dir)
    cleanup_results(out_dir)
    
    device = torch.device("cpu") # 改为 cpu 保证兼容性，可按需改为 cuda/mps
    
    test_data_path = "/Users/qishu/Desktop/一次性/研一上/矩阵理论汇报/CNN_pytorch_adversarial_attack_Fashion_MNIST/test/Fashion-MNIST"
    images = load_images(root=test_data_path)

    for model_info in models_to_evaluate:
        model_name = model_info["name"]
        model_path = model_info["path"]
        
        try:
            model = load_model(model_path, model_name, device)
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        for attack in attack_methods:
            attack_type = attack["type"]
            params = attack["params"]
            
            print(f"\n>>> Running {attack_type} attack on {model_name}...")
            
            # 统计变量
            results_stats = []
            total_l2 = 0.0
            
            # 不同方法保存在不同文件夹下: saved/adversarial/FGSM/SimpleCNN/
            # 在下面会按类别进一步分子目录
            method_base_dir = os.path.join(out_dir, attack_type, model_name)
            ensure_dir(method_base_dir)

            csv_path = os.path.join(out_dir, f"attack_results_{model_name}_{attack_type}.csv")
            
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image_path", "label_name", "label_id",
                    "pred_before_id", "pred_before_name",
                    "pred_after_id", "pred_after_name",
                    "success", "epsilon", "attack_type", "adv_image_path"
                ])
                writer.writeheader()

                for fp, tensor, label_id, label_name in tqdm(images, desc=f"{attack_type}"):
                    if attack_type == "I-FGSM":
                        res = attack_one_iter(model, tensor, label_id, epsilon, 
                                             params["iters"], params["alpha"], device)
                    else:
                        res = attack_one(model, tensor, label_id, epsilon, device)
                    
                    pred_before_id = res["pred_before"]
                    pred_after_id = res["pred_after"]
                    pred_before_name = FASHION_MNIST_LABELS[pred_before_id]
                    pred_after_name = FASHION_MNIST_LABELS[pred_after_id]
                    success = res["success"]

                    # 统计
                    is_clean_correct = (pred_before_id == label_id) if label_id is not None else False
                    is_adv_correct = (pred_after_id == label_id) if label_id is not None else False
                    
                    l2_dist = torch.norm(res["adv_image"] - tensor).item()
                    total_l2 += l2_dist

                    results_stats.append({
                        "clean_correct": is_clean_correct,
                        "adv_correct": is_adv_correct
                    })

                    # 保存图片：按类别分文件夹，并使用 [ModelName]_[ImageID]_adv.png 格式
                    category_dir = os.path.join(method_base_dir, label_name if label_name else "Unknown")
                    ensure_dir(category_dir)
                    
                    base = os.path.splitext(os.path.basename(fp))[0]
                    adv_filename = f"{model_name}_{base}_adv.png"
                    adv_fp = os.path.join(category_dir, adv_filename)
                    save_image(res["adv_image"], adv_fp)

                    writer.writerow({
                        "image_path": fp,
                        "label_name": label_name,
                        "label_id": label_id if label_id is not None else "",
                        "pred_before_id": pred_before_id,
                        "pred_before_name": pred_before_name,
                        "pred_after_id": pred_after_id,
                        "pred_after_name": pred_after_name,
                        "success": success,
                        "epsilon": epsilon,
                        "attack_type": attack_type,
                        "adv_image_path": adv_fp
                    })
                    
            calculate_and_print_metrics(model_name, attack_type, results_stats, total_l2)


if __name__ == "__main__":
    main()
