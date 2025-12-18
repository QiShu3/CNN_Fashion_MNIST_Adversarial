## 目标
- 新增 I-FGSM/PGD 迭代式攻击能力，并支持与现有 FGSM 的并排对比展示，便于直观评估“是否真的有变化”。

## 后端改动（fastapi_app.py）
- 新增迭代攻击实现（调用 quick_fgsm_attack 中的迭代函数）：
  - `iterative_fgsm_attack(image, epsilon, iters, alpha, mask=True, random_start=False)`：每步计算梯度、按 `alpha` 更新，投影到 `L_inf` 球并 `clamp(0,1)`；`mask=True` 保持背景干净。
  - `attack_one_iter(model, image, label_id, epsilon, iters, alpha, device, mask=True)`：返回结构与现有 `attack_one` 一致。
- 扩展 `/attack` 入参：
  - 新增 `attack_type: str = Form("fgsm")`（可选值：`fgsm`、`ifgsm`）。
  - 新增 `iters: int = Form(10)`、`alpha: float = Form(None)`（若为空，默认 `epsilon/iters`）。
  - 保持现有返回模型 `AttackResponse` 不变。
- 新增对比接口 `/attack/compare`：
  - 入参：`image`、`model`、`epsilon`、`iters`、`alpha`、`use_mask`。
  - 返回：同时包含 FGSM 与 I-FGSM 两套结果（预测前后 ID/名称、两张 Base64 图片）。
  - 响应模型：`AttackCompareResponse`（两个 `AttackResponse` 子对象或字段对）。
- 为新增函数与接口添加中文函数级注释。

## 前端改动（web/）
- `index.html`：
  - 新增“攻击类型”选择（`FGSM`/`I-FGSM`）、`iters`、`alpha` 输入；
  - 新增“对比攻击”按钮。
- `app.js`：
  - 根据选择调用 `/attack` 或 `/attack/compare`；
  - 对 `alpha/iters/epsilon` 做基本合法性校验；
  - 渲染两列并排结果（原图/FGSM/I-FGSM）。
- 不新增依赖，沿用现有样式。

## 默认与参数
- 默认：`iters=10`、`alpha=epsilon/iters`、`mask=True`、`random_start=False`。
- 仍支持 `epsilon` 单步 FGSM；当 `attack_type=ifgsm` 时使用迭代逻辑。

## 校验与效果对比
- 以同一图片在同一 `epsilon` 下对比 FGSM 与 I-FGSM：
  - 观察攻击成功率与视觉差异（I-FGSM 通常成功率更高、纹理更平滑）。
- 以固定成功目标，对比两者的“最小 epsilon”（可与现有 `/attack/auto` 搭配）。

## 兼容与安全
- 不更改已有接口的默认行为；
- 全流程 `clamp(0,1)` 与 `L_inf` 投影；
- 保持背景掩膜选项以确保无噪点回归。

若确认，我将按以上方案实现、联调并在本地页面展示对比效果。