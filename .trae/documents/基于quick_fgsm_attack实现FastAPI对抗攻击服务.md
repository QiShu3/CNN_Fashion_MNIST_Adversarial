## 目标
- 创建 FastAPI 应用，封装 FGSM 攻击能力，提供上传图片和模型的服务接口，返回攻击结果、原图与攻击样本。

## 端口与启动
- 使用 `uvicorn` 启动，监听 `0.0.0.0:8015`。

## 依赖
- 安装：`fastapi`、`uvicorn`、`python-multipart`（文件上传）、复用现有 `torch`、`torchvision`、`Pillow`。

## 代码结构
- 新增 `fastapi_app.py`：
  - 复用 `quick_fgsm_attack.py` 中的 `Network`、`load_model`、`fgsm_attack`、`attack_one`（通过 import），避免重复逻辑。
  - 定义请求模型与响应模型（Pydantic），并添加函数级注释。
  - 提供三个接口：
    1. `POST /attack`：接收表单上传的 `model`（可选）与 `image`，参数 `epsilon`（默认 `0.15`）；返回：
       - `success`、`pred_before_id/name`、`pred_after_id/name`、`epsilon`
       - `original_image_base64`、`adv_image_base64`
    2. `POST /attack/path`：接收 JSON 的 `model_path` 与 `image_path`，用于本地路径快速测试；同样返回上述字段。
    3. `GET /health`：健康检查。
    4. `GET /test-loop`：从 `pictures` 目录随机选择图片，使用 `model1.pth` 执行 3 次攻击，返回汇总结果数组，用于验证后端稳定性。

## 处理流程
- 图片处理：将上传或路径读取的图片统一转换为灰度 28x28、张量形状 `[1,1,28,28]`。
- 模型加载：优先 `torch.load(weights_only=False)`，失败则回退 `state_dict`；缓存模型以减少重复加载。
- 攻击执行：使用 `attack_one`，无真实标签时以模型预测作为伪标签，确保梯度非空。
- 响应图片：将原图和对抗图保存为内存 PNG，并以 Base64 编码返回（Data URI 格式）。

## 测试与验证
- 启动服务：`uvicorn fastapi_app:app --host 0.0.0.0 --port 8015`
- 验证接口：
  - 运行 `GET /health` 返回 200。
  - 运行 `GET /test-loop` 连续 3 次返回正常，后端无错误日志。

## 交付物
- 新增 `fastapi_app.py` 源码（含函数级注释）。
- 运行与验证说明（README 补充或简要步骤）。
- 启动服务并完成 3 次循环的稳定性验证。