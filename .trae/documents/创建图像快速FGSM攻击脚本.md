## 目标
- 新增脚本 `quick_fgsm_attack.py`，从图片目录与已训练模型中，执行FGSM攻击并输出对比结果与攻击样本。

## 输入输出
- 输入参数：
  - `--images`：图片根目录（如 `pictures/Fashion-MNIST-sample`）
  - `--model`：模型文件路径（如 `model1.pth`）
  - `--epsilon`：攻击强度，默认 `0.15`
  - `--out`：输出目录，默认 `saved/adversarial`
- 输出：
  - 终端打印每张图片的：原预测、攻击后预测、是否攻击成功
  - 保存对抗样本 PNG：与原图同名，后缀 `_adv_e{epsilon}.png`
  - 汇总CSV：`attack_results.csv`（包含原图路径、类别名/标签、攻击前后预测、是否成功、epsilon）

## 核心逻辑
- 加载模型：
  - 在脚本中定义与训练一致的 `Network` 类
  - 首选 `torch.load(model_path, weights_only=False)` 加载整模型
  - 失败时回退：新建 `Network()` 并 `load_state_dict(torch.load(model_path, weights_only=True))`
- 加载图片：
  - 递归读取 PNG，转灰度并缩放到 `[0,1]`，形状 `[1,1,28,28]`
  - 若目录名在 Fashion-MNIST 类别映射中（`['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']`），记录其标签id；否则标签设为 `None`
- 预测与攻击：
  - 先前向得到 `pred_before`
  - 损失选择：
    - 有真实标签：`F.cross_entropy(logits, label)`
    - 无真实标签：使用 `pred_before` 作为伪标签做无监督FGSM
  - 生成梯度并调用 `fgsm_attack(image, epsilon, grad)`，得到 `adv_image`
  - 再预测得到 `pred_after`，判断是否成功（`pred_after != pred_before`）
- 保存结果：
  - 对抗样本保存至 `--out` 目录
  - 记录CSV与控制台摘要

## 函数设计（含注释）
- `load_model(path)`: 加载并返回模型（支持两种加载方式）
- `load_images(root)`: 遍历并返回图片张量与元信息（原路径、类别名/标签）
- `fgsm_attack(image, epsilon, data_grad)`: 基本FGSM实现
- `attack_one(model, image, label, epsilon)`: 单张图片的攻击与前后预测对比
- `main(args)`: 参数解析、批量处理与结果汇总

## 使用步骤
- `source .venv/bin/activate`
- `python quick_fgsm_attack.py --images pictures/Fashion-MNIST-sample --model model1.pth --epsilon 0.15 --out saved/adversarial`

## 依赖与兼容
- 依赖：`torch`、`torchvision`、`pillow`（随已安装包可用）
- 与现有代码一致：网络结构、损失函数与FGSM定义保持一致；不修改现有文件。

## 预期效果
- 批量输出每类样本的攻击结果，生成可视化对抗样本与CSV汇总；准确率下降、预测标签变化直观可见。