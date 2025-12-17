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


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """FGSM攻击：按梯度符号扰动图像并裁剪到[0,1]"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


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

    adv_image = fgsm_attack(image, epsilon, data_grad)
    logits_adv = model(adv_image)
    pred_after = logits_adv.argmax(dim=1).item()
    success = (pred_after != pred_before) if label_id is None else (pred_after != label_id)

    return {
        "pred_before": pred_before,
        "pred_after": pred_after,
        "success": success,
        "adv_image": adv_image.detach().cpu()
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

