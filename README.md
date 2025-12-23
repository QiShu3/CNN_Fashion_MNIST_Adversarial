# CNN Fashion-MNIST Adversarial Attack

这是一个基于 PyTorch 的对抗攻击演示平台，针对 Fashion-MNIST 数据集训练的 CNN 模型进行攻击。
项目集成了 FastAPI 后端和现代化的 Web 前端界面，支持 FGSM、I-FGSM 等多种攻击方式，并提供可视化的攻击效果对比。

## ✨ 主要功能

- **多模式攻击支持**：
  - **FGSM (Fast Gradient Sign Method)**：经典的单步对抗攻击。
  - **I-FGSM (Iterative FGSM)**：迭代式攻击，攻击力更强。
  - **Targeted / Untargeted**：支持无目标攻击（误导分类）和靶向攻击（误导为指定类别）。
- **可视化交互界面**：
  - 实时上传图片进行攻击测试。
  - 动态调节 `epsilon`、迭代次数等超参数。
  - 实时查看原始图片、对抗样本及扰动噪声的可视化结果。
  - **自动攻击**：自动寻找使模型误判的最小扰动值。
  - **对比模式**：一键对比 FGSM vs I-FGSM 或 无目标 vs 靶向攻击的效果。
- **灵活的模型支持**：默认内置预训练模型，也支持用户上传自定义 `.pth` 模型。
- **RESTful API**：提供完整的后端 API 支持，方便集成。

## 🛠️ 安装与运行

本项目使用 `uv` 进行依赖管理，确保环境清洁高效。

### 1. 克隆仓库
```bash
git clone https://github.com/QiShu3/CNN_Fashion_MNIST_Adversarial.git
cd CNN_Fashion_MNIST_Adversarial
```

### 2. 安装依赖
确保已安装 [uv](https://github.com/astral-sh/uv)，然后运行：
```bash
uv sync
```

### 3. 启动服务
运行以下命令启动后端 API 服务：
```bash
uv run uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问界面
服务启动后，打开浏览器访问：
[http://localhost:8000/web/index.html](http://localhost:8000/web/index.html)

## 📂 项目结构

```
.
├── fastapi_app.py        # FastAPI 后端入口
├── quick_fgsm_attack.py  # 核心对抗攻击逻辑实现
├── web/                  # 前端静态资源
│   ├── index.html        # 主界面
│   ├── app.js            # 前端逻辑
│   └── style.css         # 样式表
├── model1.pth            # 预训练的 CNN 模型权重
├── pyproject.toml        # 项目依赖配置
├── uv.lock               # 依赖锁定文件
└── ...
```

## 📊 实验结果 (Reference)

原始模型在 Fashion-MNIST 测试集上的准确率约为 **89.22%**。
典型的 FGSM 攻击效果（无目标）：

| Epsilon | Test Accuracy |
| :--- | :--- |
| 0 | 89.22% |
| 0.05 | 35.95% |
| 0.1 | 16.63% |
| 0.15 | 7.84% |
| 0.2 | 3.61% |
| 0.3 | 1.81% |


## 📝 License
MIT
