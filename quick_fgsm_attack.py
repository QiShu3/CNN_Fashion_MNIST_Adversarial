import os
import argparse
import csv
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image


class Network(nn.Module):
    """与训练一致的CNN结构，用于加载/推理与攻击"""
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


FASHION_MNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
LABEL_NAME_TO_ID = {name: idx for idx, name in enumerate(FASHION_MNIST_LABELS)}


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """加载训练好的模型；优先整模型加载，失败则回退state_dict"""
    try:
        model = torch.load(model_path, weights_only=False, map_location=device)
        model.eval()
        return model
    except Exception:
        state = torch.load(model_path, weights_only=True, map_location=device)
        model = Network().to(device)
        model.load_state_dict(state)
        model.eval()
        return model


def load_images(root: str) -> List[Tuple[str, torch.Tensor, Optional[int], Optional[str]]]:
    """递归加载PNG图片为张量；返回 (路径, 张量, 标签id或None, 标签名或None)"""
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # [0,1], shape [1,28,28]
    ])
    results: List[Tuple[str, torch.Tensor, Optional[int], Optional[str]]] = []
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


def main():
    """参数解析、批量攻击与结果保存"""
    parser = argparse.ArgumentParser(description="快速对图片执行FGSM攻击并对比预测")
    parser.add_argument("--images", required=True, help="图片根目录")
    parser.add_argument("--model", required=True, help="训练后模型路径 .pth")
    parser.add_argument("--epsilon", type=float, default=0.15, help="FGSM攻击强度")
    parser.add_argument("--out", default="saved/adversarial", help="输出目录")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = load_model(args.model, device)

    images = load_images(args.images)
    ensure_dir(args.out)

    csv_path = os.path.join(args.out, "attack_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "label_name", "label_id",
            "pred_before_id", "pred_before_name",
            "pred_after_id", "pred_after_name",
            "success", "epsilon", "adv_image_path"
        ])
        writer.writeheader()

        for fp, tensor, label_id, label_name in images:
            res = attack_one(model, tensor, label_id, args.epsilon, device)
            pred_before_id = res["pred_before"]
            pred_after_id = res["pred_after"]
            pred_before_name = FASHION_MNIST_LABELS[pred_before_id]
            pred_after_name = FASHION_MNIST_LABELS[pred_after_id]
            success = res["success"]

            base = os.path.splitext(os.path.basename(fp))[0]
            adv_fp = os.path.join(args.out, f"{base}_adv_e{args.epsilon:.2f}.png")
            save_image(res["adv_image"], adv_fp)

            print(f"[{os.path.basename(fp)}] before={pred_before_name} -> after={pred_after_name} "
                  f"({'SUCCESS' if success else 'FAIL'}) epsilon={args.epsilon}")

            writer.writerow({
                "image_path": fp,
                "label_name": label_name,
                "label_id": label_id if label_id is not None else "",
                "pred_before_id": pred_before_id,
                "pred_before_name": pred_before_name,
                "pred_after_id": pred_after_id,
                "pred_after_name": pred_after_name,
                "success": success,
                "epsilon": args.epsilon,
                "adv_image_path": adv_fp
            })


if __name__ == "__main__":
    main()
