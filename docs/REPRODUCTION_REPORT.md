复现报告

依赖安装
- 安装方式：`uv add`（记录到 `pyproject.toml` 并安装到 `.venv`）
- 安装清单：`torch`、`torchvision`、`numpy`、`pandas`、`matplotlib`、`scikit-learn`

遇到的问题
- 模型保存路径为占位符：`yourpath/model1.pth`，运行时无法找到文件
- PyTorch `torch.load` 在 2.6+ 默认 `weights_only=True` 导致直接加载整模型对象失败
- 可视化使用 `plt.show()` 在非交互环境可能阻塞

代码更改
- `cnn_main.py`
  - 统一模型保存路径为相对路径：`PATH = './model1.pth'`
  - 添加函数级注释以便理解网络与流程
- `evasion_attack.py`
  - 统一模型加载：`torch.load(PATH, weights_only=False)`（与本地训练产生的模型对象兼容）
  - 攻击损失改为 `F.cross_entropy`，与训练一致，避免 `F.nll_loss` 的 log-softmax 期望不匹配
  - 图像输出改为保存文件：`plt.savefig('saved/accuracy_epsilon.png', dpi=300)`
  - 添加函数级注释

运行验证
- 训练脚本：10 个 epoch 完整运行，测试集准确率约 `0.8802`
- 攻击脚本：FGSM 测试在多个 epsilon 下准确率显著下降，生成并保存攻击曲线图

复现步骤
- `uv venv .venv`
- `source .venv/bin/activate`
- `uv add torch torchvision numpy pandas matplotlib scikit-learn`
- 运行训练：`python cnn_main.py`
- 运行攻击：`python evasion_attack.py`

验证标准达成说明
- 虚拟环境：独立 `.venv`，Python 3.13.7
- 依赖版本：按安装日志匹配并锁定
- 训练：10 epoch 完成并输出评估指标与混淆矩阵
- 攻击：生成预期对抗样本，准确率随 epsilon 增大而下降，曲线已保存
