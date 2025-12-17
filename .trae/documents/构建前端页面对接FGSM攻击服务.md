## 页面目标
- 通过浏览器上传图片与可选模型，设置 `epsilon`，调用 FastAPI 的 `/attack`、`/attack/adv-png` 等接口。
- 展示原图与对抗图、预测前后类别与攻击成功标记；支持一键下载对抗样本。
- 纯原生 HTML/CSS/JS，函数级注释，避免引入前端框架。

## 改动概览
- 新增 `web/` 目录：`index.html`、`style.css`、`app.js`。
- 在 `fastapi_app.py` 挂载静态资源以同源访问，避免 CORS 问题；可选开启 CORS 用于外部前端调试。

## 具体实现
### 1) 新增前端文件
- 位置：`web/index.html`、`web/style.css`、`web/app.js`
- 功能：
  - 文件选择：`image` 必选，`model` 可选；`epsilon` 滑块与数值输入联动。
  - 操作按钮：`执行攻击`、`下载对抗样本`。
  - 结果区：原图/对抗图并排预览；预测前后类别与是否成功。

#### index.html
```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>FGSM 对抗攻击演示</title>
    <link rel="stylesheet" href="./style.css" />
  </head>
  <body>
    <header>
      <h1>FGSM 对抗攻击演示</h1>
    </header>
    <main>
      <section class="panel">
        <div class="field">
          <label>上传图片（PNG/JPG）</label>
          <input id="imageInput" type="file" accept="image/*" />
        </div>
        <div class="field">
          <label>上传模型（可选，.pth）</label>
          <input id="modelInput" type="file" accept=".pth" />
        </div>
        <div class="field">
          <label>epsilon</label>
          <div class="epsilon-row">
            <input id="epsilonRange" type="range" min="0" max="0.3" step="0.01" value="0.15" />
            <input id="epsilonNumber" type="number" min="0" max="0.3" step="0.01" value="0.15" />
          </div>
        </div>
        <div class="actions">
          <button id="attackBtn">执行攻击</button>
          <button id="downloadBtn" class="secondary">下载对抗样本</button>
        </div>
      </section>

      <section class="result">
        <div class="cards">
          <div class="card">
            <h3>原始图片</h3>
            <img id="origImg" alt="原始图片预览" />
          </div>
          <div class="card">
            <h3>对抗图片</h3>
            <img id="advImg" alt="对抗图片预览" />
          </div>
        </div>
        <div class="summary">
          <div><strong>攻击结果：</strong> <span id="successText">—</span></div>
          <div><strong>预测前：</strong> <span id="predBefore">—</span></div>
          <div><strong>预测后：</strong> <span id="predAfter">—</span></div>
        </div>
      </section>
    </main>

    <footer>
      <small>后端：FastAPI (`/attack`, `/attack/adv-png`)</small>
    </footer>

    <script src="./app.js"></script>
  </body>
</html>
```

#### style.css
```css
*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;color:#222;background:#f7f7f9}
header{padding:16px 24px;background:#111;color:#fff}h1{margin:0;font-size:20px}
main{padding:24px}
.panel{background:#fff;border:1px solid #e6e6ef;border-radius:12px;padding:16px;max-width:900px;margin:0 auto}
.field{margin-bottom:16px}
.field label{display:block;margin-bottom:8px;font-weight:600}
.epsilon-row{display:flex;gap:12px;align-items:center}
.actions{display:flex;gap:12px;margin-top:8px}
button{padding:10px 16px;border:1px solid #1f6feb;background:#2da44e;color:#fff;border-radius:8px;cursor:pointer}
button.secondary{background:#0a84ff;border-color:#0a84ff}
button:disabled{opacity:.6;cursor:not-allowed}
.result{max-width:900px;margin:20px auto 0}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#fff;border:1px solid #e6e6ef;border-radius:12px;padding:12px}
.card img{width:100%;height:auto;display:block;border-radius:8px;border:1px solid #eee;background:#fafafa}
.summary{margin-top:12px;background:#fff;border:1px solid #e6e6ef;border-radius:12px;padding:12px;line-height:1.8}
footer{padding:16px 24px;color:#666}
```

#### app.js
```javascript
const API_BASE = ""; // 与 FastAPI 同源部署时留空；若跨域可改为完整地址

/**
 * 初始化事件绑定与输入联动
 */
function initPage() {
  const epsilonRange = document.getElementById("epsilonRange");
  const epsilonNumber = document.getElementById("epsilonNumber");
  const imageInput = document.getElementById("imageInput");
  const attackBtn = document.getElementById("attackBtn");
  const downloadBtn = document.getElementById("downloadBtn");

  epsilonRange.addEventListener("input", () => {
    epsilonNumber.value = epsilonRange.value;
  });
  epsilonNumber.addEventListener("input", () => {
    epsilonRange.value = epsilonNumber.value;
  });

  imageInput.addEventListener("change", previewLocalImage);
  attackBtn.addEventListener("click", handleAttack);
  downloadBtn.addEventListener("click", downloadAdversarialPng);
}

/**
 * 预览本地选择的原始图片
 */
function previewLocalImage() {
  const file = document.getElementById("imageInput").files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = document.getElementById("origImg");
  img.src = url;
}

/**
 * 调用后端 /attack 接口并渲染结果
 */
async function handleAttack() {
  const imageFile = document.getElementById("imageInput").files?.[0];
  const modelFile = document.getElementById("modelInput").files?.[0] || null;
  const epsilon = parseFloat(document.getElementById("epsilonNumber").value || "0.15");

  if (!imageFile) {
    alert("请先选择图片");
    return;
  }

  const fd = new FormData();
  fd.append("image", imageFile);
  if (modelFile) fd.append("model", modelFile);
  fd.append("epsilon", String(epsilon));

  setLoading(true);
  try {
    const resp = await fetch(`${API_BASE}/attack`, {
      method: "POST",
      body: fd,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    renderAttackResult(data);
  } catch (err) {
    console.error(err);
    alert("请求失败，请检查后端是否运行并查看控制台日志");
  } finally {
    setLoading(false);
  }
}

/**
 * 根据 /attack 响应渲染页面结果
 */
function renderAttackResult(data) {
  document.getElementById("origImg").src = data.original_image_base64;
  document.getElementById("advImg").src = data.adv_image_base64;
  document.getElementById("successText").textContent = data.success ? "成功" : "失败";
  document.getElementById("predBefore").textContent = `${data.pred_before_name} (#${data.pred_before_id})`;
  document.getElementById("predAfter").textContent = `${data.pred_after_name} (#${data.pred_after_id})`;
}

/**
 * 下载后端生成的对抗样本 PNG
 */
async function downloadAdversarialPng() {
  const imageFile = document.getElementById("imageInput").files?.[0];
  const modelFile = document.getElementById("modelInput").files?.[0] || null;
  const epsilon = parseFloat(document.getElementById("epsilonNumber").value || "0.15");

  if (!imageFile) {
    alert("请先选择图片");
    return;
  }

  const fd = new FormData();
  fd.append("image", imageFile);
  if (modelFile) fd.append("model", modelFile);
  fd.append("epsilon", String(epsilon));

  setLoading(true);
  try {
    const resp = await fetch(`${API_BASE}/attack/adv-png`, {
      method: "POST",
      body: fd,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `adv_e${epsilon}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error(err);
    alert("下载失败，请检查后端是否运行并查看控制台日志");
  } finally {
    setLoading(false);
  }
}

/**
 * 切换按钮禁用状态显示加载中
 */
function setLoading(loading) {
  const attackBtn = document.getElementById("attackBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  attackBtn.disabled = loading;
  downloadBtn.disabled = loading;
}

// 页面初始化
window.addEventListener("DOMContentLoaded", initPage);
```

### 2) 挂载静态资源与可选 CORS
- 在 `fastapi_app.py` 顶部新增：
```python
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
```
- 启用同源静态目录（将 `web/` 挂到根路径）：
```python
app.mount("/", StaticFiles(directory="web", html=True), name="web")
```
- 若需跨域（例如前端独立起一个静态服务器），可加：
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 运行与验证
- 依赖已在 `pyproject.toml`（`fastapi`、`uvicorn`、`python-multipart` 等）。如需补充，使用：`uv add <包名>`。
- 启动后端：`uv run uvicorn fastapi_app:app --reload`。
- 访问：`http://localhost:8000/`，选择 `pictures/Fashion-MNIST-sample/` 下任意 PNG 图片，设置 `epsilon`，点击“执行攻击”。
- 预期：原图与对抗图显示，预测前后类别更新；“下载对抗样本”可保存 PNG。

## 注意事项
- 后端会自动将图片转为 28×28 灰度，前端无需处理；模型上传为可选，不上传则用 `model1.pth`。
- 页面与后端同源时无需 CORS；若分离部署请按需启用 CORS。
- 仅使用原生 JS，函数级注释已在 `app.js` 中提供。