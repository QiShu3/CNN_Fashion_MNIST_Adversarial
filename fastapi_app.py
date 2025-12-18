import os
import io
import base64
import random
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from quick_fgsm_attack import (
    Network,
    load_model,
    fgsm_attack,
    fgsm_targeted_attack,
    iterative_targeted_fgsm_attack,
    attack_one,
    attack_one_iter,
    attack_one_targeted,
    attack_one_iter_targeted,
    FASHION_MNIST_LABELS,
)
import sys


app = FastAPI(title="FGSM Attack Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/web", StaticFiles(directory="web"), name="web")


def tensor_to_base64_png(t: torch.Tensor) -> str:
    """将单通道张量转换为PNG的Base64字符串"""
    img = transforms.ToPILImage()(t.squeeze(0).cpu())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def bytes_to_tensor(img_bytes: bytes) -> torch.Tensor:
    """读取图片字节并转换为模型输入张量 [1,1,28,28]"""
    pil = Image.open(io.BytesIO(img_bytes)).convert("L")
    tfm = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    return tfm(pil).unsqueeze(0)

def load_model_safe(model_path: str, device: torch.device) -> nn.Module:
    """安全加载模型文件，优先整模型加载，并允许Network类反序列化"""
    try:
        sys.modules.setdefault("__main__", sys.modules.get("__main__"))
        setattr(sys.modules["__main__"], "Network", Network)
        torch.serialization.add_safe_globals([getattr(sys.modules["__main__"], "Network")])
        mdl = torch.load(model_path, weights_only=False, map_location=device)
        mdl.eval()
        return mdl
    except Exception:
        try:
            state = torch.load(model_path, weights_only=True, map_location=device)
            mdl = Network().to(device)
            mdl.load_state_dict(state)
            mdl.eval()
            return mdl
        except Exception as e:
            raise RuntimeError(f"failed to load model from {model_path}: {e}")


class AttackResponse(BaseModel):
    success: bool
    epsilon: float
    pred_before_id: int
    pred_before_name: str
    pred_after_id: int
    pred_after_name: str
    original_image_base64: str
    adv_image_base64: str
    mask_image_base64: str


@app.get("/health")
def health():
    """健康检查接口"""
    return {"status": "ok"}


@app.post("/attack", response_model=AttackResponse)
async def attack_endpoint(
    image: UploadFile = File(...),
    model: Optional[UploadFile] = File(None),
    epsilon: float = Form(0.15),
    attack_type: str = Form("fgsm"),
    iters: int = Form(10),
    alpha: Optional[float] = Form(None),
    use_mask: bool = Form(True),
):
    """接收上传的模型与图片，执行攻击（FGSM 或 I-FGSM）并返回结果与样本"""
    device = torch.device("cpu")
    if model is not None:
        model_bytes = await model.read()
        try:
            sys.modules.setdefault("__main__", sys.modules.get("__main__"))
            setattr(sys.modules["__main__"], "Network", Network)
            torch.serialization.add_safe_globals([getattr(sys.modules["__main__"], "Network")])
            mdl = torch.load(io.BytesIO(model_bytes), weights_only=False, map_location=device)
            mdl.eval()
        except Exception:
            state = torch.load(io.BytesIO(model_bytes), weights_only=True, map_location=device)
            mdl = Network().to(device)
            mdl.load_state_dict(state)
            mdl.eval()
    else:
        mdl = load_model_safe("./model1.pth", device)

    img_bytes = await image.read()
    tensor = bytes_to_tensor(img_bytes)
    if attack_type.lower() in ("ifgsm", "pgd"):
        res = attack_one_iter(mdl, tensor, None, epsilon, iters, alpha, device, use_mask=use_mask)
    else:
        res = attack_one(mdl, tensor, None, epsilon, device)

    pred_before_id = res["pred_before"]
    pred_after_id = res["pred_after"]
    original_b64 = tensor_to_base64_png(tensor)
    adv_b64 = tensor_to_base64_png(res["adv_image"])
    mask_b64 = tensor_to_base64_png(res["mask"])

    return AttackResponse(
        success=res["success"],
        epsilon=epsilon,
        pred_before_id=pred_before_id,
        pred_before_name=FASHION_MNIST_LABELS[pred_before_id],
        pred_after_id=pred_after_id,
        pred_after_name=FASHION_MNIST_LABELS[pred_after_id],
        original_image_base64=original_b64,
        adv_image_base64=adv_b64,
        mask_image_base64=mask_b64,
    )

class AttackCompareResponse(BaseModel):
    success_fgsm: bool
    success_ifgsm: bool
    epsilon: float
    pred_before_id: int
    pred_before_name: str
    pred_after_fgsm_id: int
    pred_after_fgsm_name: str
    pred_after_ifgsm_id: int
    pred_after_ifgsm_name: str
    original_image_base64: str
    adv_fgsm_base64: str
    adv_ifgsm_base64: str
    mask_fgsm_base64: str
    mask_ifgsm_base64: str

@app.post("/attack/compare", response_model=AttackCompareResponse)
async def attack_compare_endpoint(
    image: UploadFile = File(...),
    model: Optional[UploadFile] = File(None),
    epsilon: float = Form(0.15),
    iters: int = Form(10),
    alpha: Optional[float] = Form(None),
    use_mask: bool = Form(True),
):
    """对比同一图片在相同 epsilon 下的 FGSM 与 I-FGSM 攻击结果"""
    device = torch.device("cpu")
    if model is not None:
        model_bytes = await model.read()
        try:
            sys.modules.setdefault("__main__", sys.modules.get("__main__"))
            setattr(sys.modules["__main__"], "Network", Network)
            torch.serialization.add_safe_globals([getattr(sys.modules["__main__"], "Network")])
            mdl = torch.load(io.BytesIO(model_bytes), weights_only=False, map_location=device)
            mdl.eval()
        except Exception:
            state = torch.load(io.BytesIO(model_bytes), weights_only=True, map_location=device)
            mdl = Network().to(device)
            mdl.load_state_dict(state)
            mdl.eval()
    else:
        mdl = load_model_safe("./model1.pth", device)

    img_bytes = await image.read()
    tensor = bytes_to_tensor(img_bytes)
    res_fgsm = attack_one(mdl, tensor, None, epsilon, device)
    res_ifgsm = attack_one_iter(mdl, tensor, None, epsilon, iters, alpha, device, use_mask=use_mask)

    pred_before_id = res_fgsm["pred_before"]
    original_b64 = tensor_to_base64_png(tensor)
    adv_fgsm_b64 = tensor_to_base64_png(res_fgsm["adv_image"])
    adv_ifgsm_b64 = tensor_to_base64_png(res_ifgsm["adv_image"])
    mask_fgsm_b64 = tensor_to_base64_png(res_fgsm["mask"])
    mask_ifgsm_b64 = tensor_to_base64_png(res_ifgsm["mask"])

    return AttackCompareResponse(
        success_fgsm=res_fgsm["success"],
        success_ifgsm=res_ifgsm["success"],
        epsilon=epsilon,
        pred_before_id=pred_before_id,
        pred_before_name=FASHION_MNIST_LABELS[pred_before_id],
        pred_after_fgsm_id=res_fgsm["pred_after"],
        pred_after_fgsm_name=FASHION_MNIST_LABELS[res_fgsm["pred_after"]],
        pred_after_ifgsm_id=res_ifgsm["pred_after"],
        pred_after_ifgsm_name=FASHION_MNIST_LABELS[res_ifgsm["pred_after"]],
        original_image_base64=original_b64,
        adv_fgsm_base64=adv_fgsm_b64,
        adv_ifgsm_base64=adv_ifgsm_b64,
        mask_fgsm_base64=mask_fgsm_b64,
        mask_ifgsm_base64=mask_ifgsm_b64,
    )

class TargetedAttackResponse(BaseModel):
    success: bool
    epsilon: float
    target_class_id: int
    target_class_name: str
    pred_before_id: int
    pred_before_name: str
    pred_after_id: int
    pred_after_name: str
    original_image_base64: str
    adv_image_base64: str
    mask_image_base64: str

@app.post("/attack/targeted", response_model=TargetedAttackResponse)
async def attack_targeted_endpoint(
    image: UploadFile = File(...),
    model: Optional[UploadFile] = File(None),
    target_class_id: int = Form(...),
    epsilon: float = Form(0.15),
    attack_type: str = Form("fgsm"),
    iters: int = Form(10),
    alpha: Optional[float] = Form(None),
    use_mask: bool = Form(True),
):
    """靶向攻击接口：尝试将图片伪装成 target_class_id"""
    device = torch.device("cpu")
    if model is not None:
        # Load uploaded model
        model_bytes = await model.read()
        try:
            sys.modules.setdefault("__main__", sys.modules.get("__main__"))
            setattr(sys.modules["__main__"], "Network", Network)
            torch.serialization.add_safe_globals([getattr(sys.modules["__main__"], "Network")])
            mdl = torch.load(io.BytesIO(model_bytes), weights_only=False, map_location=device)
            mdl.eval()
        except Exception:
            state = torch.load(io.BytesIO(model_bytes), weights_only=True, map_location=device)
            mdl = Network().to(device)
            mdl.load_state_dict(state)
            mdl.eval()
    else:
        mdl = load_model_safe("./model1.pth", device)

    img_bytes = await image.read()
    tensor = bytes_to_tensor(img_bytes)
    
    if attack_type.lower() in ("ifgsm", "pgd"):
        res = attack_one_iter_targeted(
            mdl, tensor, target_class_id, epsilon, iters, alpha, device, use_mask=use_mask
        )
    else:
        res = attack_one_targeted(
            mdl, tensor, target_class_id, epsilon, device
        )

    pred_before_id = res["pred_before"]
    pred_after_id = res["pred_after"]
    original_b64 = tensor_to_base64_png(tensor)
    adv_b64 = tensor_to_base64_png(res["adv_image"])
    mask_b64 = tensor_to_base64_png(res["mask"])

    return TargetedAttackResponse(
        success=res["success"],
        epsilon=epsilon,
        target_class_id=target_class_id,
        target_class_name=FASHION_MNIST_LABELS[target_class_id],
        pred_before_id=pred_before_id,
        pred_before_name=FASHION_MNIST_LABELS[pred_before_id],
        pred_after_id=pred_after_id,
        pred_after_name=FASHION_MNIST_LABELS[pred_after_id],
        original_image_base64=original_b64,
        adv_image_base64=adv_b64,
        mask_image_base64=mask_b64,
    )

class AttackModeCompareResponse(BaseModel):
    # 分别记录无目标和靶向各自成功的最小 epsilon
    untargeted_epsilon: float
    targeted_epsilon: float
    
    target_class_id: int
    target_class_name: str
    
    pred_before_id: int
    pred_before_name: str
    
    # Untargeted results
    untargeted_success: bool
    untargeted_pred_after_id: int
    untargeted_pred_after_name: str
    untargeted_adv_base64: str
    
    # Targeted results
    targeted_success: bool
    targeted_pred_after_id: int
    targeted_pred_after_name: str
    targeted_adv_base64: str
    
    original_image_base64: str

@app.post("/attack/compare_modes", response_model=AttackModeCompareResponse)
async def attack_compare_modes_endpoint(
    image: UploadFile = File(...),
    target_class_id: int = Form(...),
    model: Optional[UploadFile] = File(None),
    epsilon: float = Form(0.15),
    attack_type: str = Form("fgsm"), # 'fgsm' or 'ifgsm'
    iters: int = Form(10),
    alpha: Optional[float] = Form(None),
    use_mask: bool = Form(True),
):
    """
    对比无目标攻击与靶向攻击，分别搜索使得各自成功的最小 epsilon。
    为了提高效率，不再强制两者同时成功，而是分别记录各自成功的时刻。
    如果搜索到上限仍未成功，则返回上限时的结果。
    """
    device = torch.device("cpu")
    # Model loading
    if model is not None:
        model_bytes = await model.read()
        try:
            sys.modules.setdefault("__main__", sys.modules.get("__main__"))
            setattr(sys.modules["__main__"], "Network", Network)
            torch.serialization.add_safe_globals([getattr(sys.modules["__main__"], "Network")])
            mdl = torch.load(io.BytesIO(model_bytes), weights_only=False, map_location=device)
            mdl.eval()
        except Exception:
            state = torch.load(io.BytesIO(model_bytes), weights_only=True, map_location=device)
            mdl = Network().to(device)
            mdl.load_state_dict(state)
            mdl.eval()
    else:
        mdl = load_model_safe("./model1.pth", device)

    img_bytes = await image.read()
    tensor = bytes_to_tensor(img_bytes)
    
    # 搜索参数
    max_search_epsilon = max(epsilon, 0.5) 
    step = 0.005
    current_eps = 0.0
    
    # 状态追踪
    found_untargeted = False
    found_targeted = False
    
    best_res_untargeted = None
    best_res_targeted = None
    
    untargeted_eps_val = max_search_epsilon
    targeted_eps_val = max_search_epsilon
    
    # 初始预测
    # 我们可以先做一次 0 扰动的预测，看看是否本身就预测错了
    # 但 attack_one 会返回 pred_before，所以直接进入循环即可。
    
    while current_eps <= max_search_epsilon + 1e-9:
        # 只要有一方还没成功，就继续跑该方的攻击
        
        # 1. 无目标攻击
        if not found_untargeted:
            if attack_type.lower() in ("ifgsm", "pgd"):
                res_u = attack_one_iter(mdl, tensor, None, current_eps, iters, alpha, device, use_mask=use_mask)
            else:
                res_u = attack_one(mdl, tensor, None, current_eps, device)
            
            best_res_untargeted = res_u
            if res_u["success"]:
                found_untargeted = True
                untargeted_eps_val = current_eps

        # 2. 靶向攻击
        if not found_targeted:
            if attack_type.lower() in ("ifgsm", "pgd"):
                res_t = attack_one_iter_targeted(mdl, tensor, target_class_id, current_eps, iters, alpha, device, use_mask=use_mask)
            else:
                res_t = attack_one_targeted(mdl, tensor, target_class_id, current_eps, device)
            
            best_res_targeted = res_t
            if res_t["success"]:
                found_targeted = True
                targeted_eps_val = current_eps
        
        # 如果两者都找到了，或者两者都无需再找（比如已经到了上限，但while循环自然会处理上限），提前退出
        if found_untargeted and found_targeted:
            break
            
        current_eps += step

    original_b64 = tensor_to_base64_png(tensor)
    
    return AttackModeCompareResponse(
        untargeted_epsilon=round(untargeted_eps_val, 6),
        targeted_epsilon=round(targeted_eps_val, 6),
        
        target_class_id=target_class_id,
        target_class_name=FASHION_MNIST_LABELS[target_class_id],
        
        pred_before_id=best_res_untargeted["pred_before"],
        pred_before_name=FASHION_MNIST_LABELS[best_res_untargeted["pred_before"]],
        
        untargeted_success=best_res_untargeted["success"],
        untargeted_pred_after_id=best_res_untargeted["pred_after"],
        untargeted_pred_after_name=FASHION_MNIST_LABELS[best_res_untargeted["pred_after"]],
        untargeted_adv_base64=tensor_to_base64_png(best_res_untargeted["adv_image"]),
        
        targeted_success=best_res_targeted["success"],
        targeted_pred_after_id=best_res_targeted["pred_after"],
        targeted_pred_after_name=FASHION_MNIST_LABELS[best_res_targeted["pred_after"]],
        targeted_adv_base64=tensor_to_base64_png(best_res_targeted["adv_image"]),
        
        original_image_base64=original_b64
    )

@app.post("/attack/adv-png")
async def attack_adv_png(
    image: UploadFile = File(...),
    model: Optional[UploadFile] = File(None),
    epsilon: float = Form(0.15),
):
    """返回对抗样本的PNG二进制数据"""
    device = torch.device("cpu")
    if model is not None:
        model_bytes = await model.read()
        try:
            sys.modules.setdefault("__main__", sys.modules.get("__main__"))
            setattr(sys.modules["__main__"], "Network", Network)
            torch.serialization.add_safe_globals([getattr(sys.modules["__main__"], "Network")])
            mdl = torch.load(io.BytesIO(model_bytes), weights_only=False, map_location=device)
            mdl.eval()
        except Exception:
            state = torch.load(io.BytesIO(model_bytes), weights_only=True, map_location=device)
            mdl = Network().to(device)
            mdl.load_state_dict(state)
            mdl.eval()
    else:
        mdl = load_model_safe("./model1.pth", device)

    img_bytes = await image.read()
    tensor = bytes_to_tensor(img_bytes)
    res = attack_one(mdl, tensor, None, epsilon, device)
    adv_b64 = tensor_to_base64_png(res["adv_image"])
    raw = base64.b64decode(adv_b64.split(",")[1])
    return StreamingResponse(io.BytesIO(raw), media_type="image/png")

@app.post("/attack/preview-png")
async def attack_preview_png(
    image: UploadFile = File(...),
    model: Optional[UploadFile] = File(None),
    epsilon: float = Form(0.15),
):
    """返回原图与对抗图的并排预览PNG"""
    device = torch.device("cpu")
    if model is not None:
        model_bytes = await model.read()
        try:
            sys.modules.setdefault("__main__", sys.modules.get("__main__"))
            setattr(sys.modules["__main__"], "Network", Network)
            torch.serialization.add_safe_globals([getattr(sys.modules["__main__"], "Network")])
            mdl = torch.load(io.BytesIO(model_bytes), weights_only=False, map_location=device)
            mdl.eval()
        except Exception:
            state = torch.load(io.BytesIO(model_bytes), weights_only=True, map_location=device)
            mdl = Network().to(device)
            mdl.load_state_dict(state)
            mdl.eval()
    else:
        mdl = load_model_safe("./model1.pth", device)

    img_bytes = await image.read()
    tensor = bytes_to_tensor(img_bytes)
    res = attack_one(mdl, tensor, None, epsilon, device)

    orig_img = transforms.ToPILImage()(tensor.squeeze(0))
    adv_img = transforms.ToPILImage()(res["adv_image"].squeeze(0))
    w, h = orig_img.size
    canvas = Image.new("L", (w * 2, h))
    canvas.paste(orig_img, (0, 0))
    canvas.paste(adv_img, (w, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/attack/auto", response_model=AttackResponse)
async def attack_auto(
    image: UploadFile = File(...),
    model: Optional[UploadFile] = File(None),
    step: float = Form(0.01),
    max_epsilon: float = Form(0.3),
):
    """自动从0开始递增epsilon直至攻击成功或达到上限，返回首个成功扰动值与结果"""
    device = torch.device("cpu")
    if step <= 0:
        step = 0.01
    if max_epsilon < 0:
        max_epsilon = 0.0
    if max_epsilon > 1.0:
        max_epsilon = 1.0

    if model is not None:
        model_bytes = await model.read()
        try:
            sys.modules.setdefault("__main__", sys.modules.get("__main__"))
            setattr(sys.modules["__main__"], "Network", Network)
            torch.serialization.add_safe_globals([getattr(sys.modules["__main__"], "Network")])
            mdl = torch.load(io.BytesIO(model_bytes), weights_only=False, map_location=device)
            mdl.eval()
        except Exception:
            state = torch.load(io.BytesIO(model_bytes), weights_only=True, map_location=device)
            mdl = Network().to(device)
            mdl.load_state_dict(state)
            mdl.eval()
    else:
        mdl = load_model_safe("./model1.pth", device)

    img_bytes = await image.read()
    tensor = bytes_to_tensor(img_bytes)

    best_res = None
    e = 0.0
    while e <= max_epsilon + 1e-12:
        res = attack_one(mdl, tensor, None, e, device)
        best_res = res
        if res["success"]:
            pred_before_id = res["pred_before"]
            pred_after_id = res["pred_after"]
            original_b64 = tensor_to_base64_png(tensor)
            adv_b64 = tensor_to_base64_png(res["adv_image"])
            mask_b64 = tensor_to_base64_png(res["mask"])
            return AttackResponse(
                success=True,
                epsilon=round(e, 6),
                pred_before_id=pred_before_id,
                pred_before_name=FASHION_MNIST_LABELS[pred_before_id],
                pred_after_id=pred_after_id,
                pred_after_name=FASHION_MNIST_LABELS[pred_after_id],
                original_image_base64=original_b64,
                adv_image_base64=adv_b64,
                mask_image_base64=mask_b64,
            )
        e += step

    pred_before_id = best_res["pred_before"]
    pred_after_id = best_res["pred_after"]
    original_b64 = tensor_to_base64_png(tensor)
    adv_b64 = tensor_to_base64_png(best_res["adv_image"])
    mask_b64 = tensor_to_base64_png(best_res["mask"])
    return AttackResponse(
        success=False,
        epsilon=round(min(max_epsilon, e - step), 6),
        pred_before_id=pred_before_id,
        pred_before_name=FASHION_MNIST_LABELS[pred_before_id],
        pred_after_id=pred_after_id,
        pred_after_name=FASHION_MNIST_LABELS[pred_after_id],
        original_image_base64=original_b64,
        adv_image_base64=adv_b64,
        mask_image_base64=mask_b64,
    )

@app.post("/attack/path", response_model=AttackResponse)
def attack_by_path(
    image_path: str,
    model_path: str = "./model1.pth",
    epsilon: float = 0.15,
):
    """通过本地文件路径加载模型与图片并执行攻击"""
    device = torch.device("cpu")
    mdl = load_model_safe(model_path, device)
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    tensor = bytes_to_tensor(img_bytes)
    res = attack_one(mdl, tensor, None, epsilon, device)

    pred_before_id = res["pred_before"]
    pred_after_id = res["pred_after"]
    original_b64 = tensor_to_base64_png(tensor)
    adv_b64 = tensor_to_base64_png(res["adv_image"])
    mask_b64 = tensor_to_base64_png(res["mask"])

    return AttackResponse(
        success=res["success"],
        epsilon=epsilon,
        pred_before_id=pred_before_id,
        pred_before_name=FASHION_MNIST_LABELS[pred_before_id],
        pred_after_id=pred_after_id,
        pred_after_name=FASHION_MNIST_LABELS[pred_after_id],
        original_image_base64=original_b64,
        adv_image_base64=adv_b64,
        mask_image_base64=mask_b64,
    )


@app.get("/test-loop")
def test_loop(epsilon: float = 0.15, loops: int = 3):
    """随机选择图片并对同一模型执行攻击多次，返回汇总结果"""
    device = torch.device("cpu")
    mdl = load_model_safe("./model1.pth", device)

    root = "./pictures"
    pngs: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".png"):
                pngs.append(os.path.join(dirpath, fn))
    if not pngs:
        return {"error": "no png images found", "root": root}

    results: List[Dict] = []
    for _ in range(loops):
        p = random.choice(pngs)
        with open(p, "rb") as f:
            img_bytes = f.read()
        tensor = bytes_to_tensor(img_bytes)
        res = attack_one(mdl, tensor, None, epsilon, device)
        pred_before_id = res["pred_before"]
        pred_after_id = res["pred_after"]
        results.append({
            "image_path": p,
            "epsilon": epsilon,
            "success": res["success"],
            "pred_before_id": pred_before_id,
            "pred_before_name": FASHION_MNIST_LABELS[pred_before_id],
            "pred_after_id": pred_after_id,
            "pred_after_name": FASHION_MNIST_LABELS[pred_after_id],
        })
    return {"count": len(results), "results": results}


# Mount static frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")
