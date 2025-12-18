const API_BASE = ""; // 与 FastAPI 同源部署时留空；若跨域可改为完整地址

let currentCompareData = null; // 存储最近一次对比攻击的结果

/**
 * 初始化事件绑定与输入联动
 */
function initPage() {
  const epsilonRange = document.getElementById("epsilonRange");
  const epsilonNumber = document.getElementById("epsilonNumber");
  const imageInput = document.getElementById("imageInput");
  const attackBtn = document.getElementById("attackBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  const autoBtn = document.getElementById("autoBtn");
  const compareBtn = document.getElementById("compareBtn");
  const compareModesBtn = document.getElementById("compareModesBtn");

  epsilonRange.addEventListener("input", () => {
    epsilonNumber.value = epsilonRange.value;
  });
  epsilonNumber.addEventListener("input", () => {
    epsilonRange.value = epsilonNumber.value;
  });

  imageInput.addEventListener("change", previewLocalImage);
  attackBtn.addEventListener("click", handleAttack);
  downloadBtn.addEventListener("click", downloadAdversarialPng);
  autoBtn.addEventListener("click", handleAutoAttack);
  compareBtn.addEventListener("click", handleCompareAttack);
  compareModesBtn.addEventListener("click", handleCompareModes);

  // 绑定对比模式切换事件
  const radios = document.getElementsByName("compareMode");
  radios.forEach(radio => {
    radio.addEventListener("change", updateCompareView);
  });
  
  // 绑定攻击目标模式切换
  const targetRadios = document.getElementsByName("targetMode");
  targetRadios.forEach(radio => {
    radio.addEventListener("change", toggleTargetInput);
  });
}

/**
 * 切换目标类别选择器的显示
 */
function toggleTargetInput() {
  const mode = document.querySelector('input[name="targetMode"]:checked').value;
  const row = document.getElementById("targetClassRow");
  if (mode === "targeted") {
    row.style.display = "block";
  } else {
    row.style.display = "none";
  }
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
  const attackType = document.getElementById("attackType").value;
  const iters = parseInt(document.getElementById("itersNumber").value || "10", 10);
  const alphaInput = document.getElementById("alphaNumber").value;
  const alpha = alphaInput ? parseFloat(alphaInput) : null;

  if (!imageFile) {
    alert("请先选择图片");
    return;
  }

  // 隐藏对比切换控件与无关卡片
  document.getElementById("compareControl").style.display = "none";
  document.getElementById("advIfgsmCard").style.display = "none";
  document.getElementById("predAfterIfgsmRow").style.display = "none";
  document.getElementById("advTargetedCard").style.display = "none";
  document.getElementById("predAfterTargetedRow").style.display = "none";

  const fd = new FormData();
  fd.append("image", imageFile);
  if (modelFile) fd.append("model", modelFile);
  fd.append("epsilon", String(epsilon));
  fd.append("attack_type", attackType);
  if (attackType === "ifgsm") {
    fd.append("iters", String(iters));
    if (alpha !== null) fd.append("alpha", String(alpha));
    fd.append("use_mask", "true");
  }

  // 检查是否为靶向攻击
  const targetMode = document.querySelector('input[name="targetMode"]:checked').value;
  const targetClass = document.getElementById("targetClassInput").value;
  let endpoint = "/attack";
  
  if (targetMode === "targeted") {
    endpoint = "/attack/targeted";
    fd.append("target_class_id", targetClass);
  }

  setLoading(true);
  try {
    const resp = await fetch(`${API_BASE}${endpoint}`, {
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
  if (data.mask_image_base64) {
    document.getElementById("maskImg").src = data.mask_image_base64;
    document.getElementById("maskImg").parentElement.style.display = "block";
  } else {
    document.getElementById("maskImg").parentElement.style.display = "none";
  }
  document.getElementById("successText").textContent = data.success ? "成功" : "失败";
  const eps = typeof data.epsilon === "number" ? data.epsilon.toFixed(3) : String(data.epsilon);
  document.getElementById("epsilonText").textContent = eps;
  document.getElementById("predBefore").textContent = `${data.pred_before_name} (#${data.pred_before_id})`;
  document.getElementById("predAfter").textContent = `${data.pred_after_name} (#${data.pred_after_id})`;
  
  // 如果是靶向攻击，显示目标信息
  if (data.target_class_id !== undefined) {
    document.getElementById("advTitle").textContent = `靶向攻击结果 (目标: ${data.target_class_name})`;
  } else {
    document.getElementById("advTitle").textContent = "对抗图片";
  }

  renderComparisons(data.original_image_base64, data.adv_image_base64);
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
  const autoBtn = document.getElementById("autoBtn");
  autoBtn.disabled = loading;
  const compareBtn = document.getElementById("compareBtn");
  compareBtn.disabled = loading;
  const compareModesBtn = document.getElementById("compareModesBtn");
  compareModesBtn.disabled = loading;
}

// 页面初始化
window.addEventListener("DOMContentLoaded", initPage);

/**
 * 调用后端 /attack/auto 自动搜索最小成功扰动
 */
async function handleAutoAttack() {
  const imageFile = document.getElementById("imageInput").files?.[0];
  const modelFile = document.getElementById("modelInput").files?.[0] || null;
  const step = parseFloat(document.getElementById("stepNumber").value || "0.01");
  const maxEps = parseFloat(document.getElementById("maxEpsNumber").value || "0.3");

  if (!imageFile) {
    alert("请先选择图片");
    return;
  }
  if (!(step > 0)) {
    alert("步长必须为正数");
    return;
  }
  if (maxEps < 0 || maxEps > 1) {
    alert("最大扰动需在 [0,1] 范围内");
    return;
  }

  const fd = new FormData();
  fd.append("image", imageFile);
  if (modelFile) fd.append("model", modelFile);
  fd.append("step", String(step));
  fd.append("max_epsilon", String(maxEps));

  setLoading(true);
  try {
    const resp = await fetch(`${API_BASE}/attack/auto`, {
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
 * 对比同一 epsilon 下的 FGSM 与 I-FGSM
 */
async function handleCompareAttack() {
  const imageFile = document.getElementById("imageInput").files?.[0];
  const modelFile = document.getElementById("modelInput").files?.[0] || null;
  const epsilon = parseFloat(document.getElementById("epsilonNumber").value || "0.15");
  const iters = parseInt(document.getElementById("itersNumber").value || "10", 10);
  const alphaInput = document.getElementById("alphaNumber").value;
  const alpha = alphaInput ? parseFloat(alphaInput) : null;
  if (!imageFile) {
    alert("请先选择图片");
    return;
  }
  const fd = new FormData();
  fd.append("image", imageFile);
  if (modelFile) fd.append("model", modelFile);
  fd.append("epsilon", String(epsilon));
  fd.append("iters", String(iters));
  if (alpha !== null) fd.append("alpha", String(alpha));
  fd.append("use_mask", "true");
  setLoading(true);
  try {
    const resp = await fetch(`${API_BASE}/attack/compare`, {
      method: "POST",
      body: fd,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    currentCompareData = data; // 保存数据以供切换
    currentCompareData.type = "fgsm_vs_ifgsm"; // 标记对比类型

    document.getElementById("origImg").src = data.original_image_base64;
    document.getElementById("advImg").src = data.adv_fgsm_base64;
    
    // 隐藏其他无关卡片
    document.getElementById("advIfgsmCard").style.display = "block";
    document.getElementById("advIfgsmImg").src = data.adv_ifgsm_base64;
    document.getElementById("advTargetedCard").style.display = "none";
    
    // 显示对比切换控件，并重置为 FGSM
    const ctrl = document.getElementById("compareControl");
    ctrl.style.display = "block";
    document.getElementById("mode1Label").textContent = "FGSM";
    document.getElementById("mode2Label").textContent = "I-FGSM";
    document.querySelector('input[name="compareMode"][value="mode1"]').checked = true;

    document.getElementById("successText").textContent =
      `FGSM: ${data.success_fgsm ? "成功" : "失败"} / I-FGSM: ${data.success_ifgsm ? "成功" : "失败"}`;
    const eps = typeof data.epsilon === "number" ? data.epsilon.toFixed(3) : String(data.epsilon);
    document.getElementById("epsilonText").textContent = `FGSM & I-FGSM: ${eps}`;
    document.getElementById("predBefore").textContent = `${data.pred_before_name} (#${data.pred_before_id})`;
    document.getElementById("predAfter").textContent = `${data.pred_after_fgsm_name} (#${data.pred_after_fgsm_id})`;
    
    const ifgsmRow = document.getElementById("predAfterIfgsmRow");
    ifgsmRow.style.display = "block";
    document.getElementById("predAfterIfgsm").textContent = `${data.pred_after_ifgsm_name} (#${data.pred_after_ifgsm_id})`;
    
    document.getElementById("predAfterTargetedRow").style.display = "none";

    if (data.mask_fgsm_base64) {
      document.getElementById("maskImg").src = data.mask_fgsm_base64;
      document.getElementById("maskImg").parentElement.style.display = "block";
    } else {
      document.getElementById("maskImg").parentElement.style.display = "none";
    }

    renderComparisons(data.original_image_base64, data.adv_fgsm_base64);
  } catch (err) {
    console.error(err);
    alert("请求失败，请检查后端是否运行并查看控制台日志");
  } finally {
    setLoading(false);
  }
}

/**
 * 对比 无目标 vs 靶向
 */
async function handleCompareModes() {
  const imageFile = document.getElementById("imageInput").files?.[0];
  const modelFile = document.getElementById("modelInput").files?.[0] || null;
  const epsilon = parseFloat(document.getElementById("epsilonNumber").value || "0.15");
  const iters = parseInt(document.getElementById("itersNumber").value || "10", 10);
  const alphaInput = document.getElementById("alphaNumber").value;
  const alpha = alphaInput ? parseFloat(alphaInput) : null;
  const attackType = document.getElementById("attackType").value;
  
  // 必须指定目标类别，否则对比无意义（默认给个0？）
  // 这里直接取 UI 上选的值，如果 UI 上选了无目标，这里还是会取 select 的值
  const targetClass = document.getElementById("targetClassInput").value;

  if (!imageFile) {
    alert("请先选择图片");
    return;
  }
  
  const fd = new FormData();
  fd.append("image", imageFile);
  if (modelFile) fd.append("model", modelFile);
  fd.append("epsilon", String(epsilon));
  fd.append("attack_type", attackType);
  fd.append("iters", String(iters));
  if (alpha !== null) fd.append("alpha", String(alpha));
  fd.append("use_mask", "true");
  fd.append("target_class_id", targetClass);

  setLoading(true);
  try {
    const resp = await fetch(`${API_BASE}/attack/compare_modes`, {
      method: "POST",
      body: fd,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    currentCompareData = data;
    currentCompareData.type = "untargeted_vs_targeted"; // 标记类型

    document.getElementById("origImg").src = data.original_image_base64;
    document.getElementById("advImg").src = data.untargeted_adv_base64;
    
    // 显示靶向结果卡片，隐藏 I-FGSM 卡片
    document.getElementById("advIfgsmCard").style.display = "none";
    document.getElementById("advTargetedCard").style.display = "block";
    document.getElementById("advTargetedImg").src = data.targeted_adv_base64;

    // 显示对比切换控件
    const ctrl = document.getElementById("compareControl");
    ctrl.style.display = "block";
    document.getElementById("mode1Label").textContent = "无目标";
    document.getElementById("mode2Label").textContent = `靶向 (目标:${data.target_class_name})`;
    document.querySelector('input[name="compareMode"][value="mode1"]').checked = true;

    document.getElementById("successText").textContent =
      `无目标: ${data.untargeted_success ? "成功" : "失败"} / 靶向: ${data.targeted_success ? "成功" : "失败"}`;
    
    // 分别显示两个 Epsilon
    const epsU = typeof data.untargeted_epsilon === "number" ? data.untargeted_epsilon.toFixed(3) : String(data.untargeted_epsilon);
    const epsT = typeof data.targeted_epsilon === "number" ? data.targeted_epsilon.toFixed(3) : String(data.targeted_epsilon);
    document.getElementById("epsilonText").textContent = `无目标:${epsU} / 靶向:${epsT}`;
    
    document.getElementById("predBefore").textContent = `${data.pred_before_name} (#${data.pred_before_id})`;
    
    document.getElementById("predAfter").textContent = `${data.untargeted_pred_after_name} (#${data.untargeted_pred_after_id})`;
    
    document.getElementById("predAfterIfgsmRow").style.display = "none";
    const targetedRow = document.getElementById("predAfterTargetedRow");
    targetedRow.style.display = "block";
    document.getElementById("predAfterTargeted").textContent = `${data.targeted_pred_after_name} (#${data.targeted_pred_after_id})`;

    // 默认显示无目标掩膜（目前后端没返回单独的掩膜，假设返回了adv_image就包含mask生成逻辑，
    // 但 attack/compare_modes 接口暂时没返回 mask base64，这需要后端补充，
    // 或者我们暂时只显示原图与对抗图的差异，不显示单独的 mask 图片
    // 检查后端代码发现确实没返回 mask_base64，这会导致 mask 显示空白或报错
    // 我们可以暂时隐藏 mask 或在后端加上 mask 返回。
    // 为了完整性，最好后端加上。但这里先做前端容错。）
    
    // 实际上后端代码里 compare_modes 没返回 mask。
    // 这是一个小缺陷。为了快速响应，我先在前端隐藏 Mask 区域或显示为空。
    document.getElementById("maskImg").parentElement.style.display = "none";

    renderComparisons(data.original_image_base64, data.untargeted_adv_base64);

  } catch (err) {
    console.error(err);
    alert("请求失败，请检查后端是否运行并查看控制台日志");
  } finally {
    setLoading(false);
  }
}

/**
 * 切换对比视图（FGSM / I-FGSM 或 无目标 / 靶向）
 */
function updateCompareView() {
  console.log("updateCompareView called");
  if (!currentCompareData) {
    console.warn("No compare data available");
    return;
  }
  const mode = document.querySelector('input[name="compareMode"]:checked').value;
  console.log("Switching to mode:", mode);

  const orig = currentCompareData.original_image_base64;
  let adv, mask;

  if (currentCompareData.type === "fgsm_vs_ifgsm") {
    // FGSM vs I-FGSM
    if (mode === "mode2") { // I-FGSM
      adv = currentCompareData.adv_ifgsm_base64;
      mask = currentCompareData.mask_ifgsm_base64;
    } else { // FGSM
      adv = currentCompareData.adv_fgsm_base64;
      mask = currentCompareData.mask_fgsm_base64;
    }
  } else if (currentCompareData.type === "untargeted_vs_targeted") {
    // Untargeted vs Targeted
    if (mode === "mode2") { // Targeted
      adv = currentCompareData.targeted_adv_base64;
      mask = null; // 后端未返回
    } else { // Untargeted
      adv = currentCompareData.untargeted_adv_base64;
      mask = null; // 后端未返回
    }
  }

  // 更新掩膜显示
  if (mask) {
    document.getElementById("maskImg").src = mask;
    document.getElementById("maskImg").parentElement.style.display = "block";
  } else {
    document.getElementById("maskImg").parentElement.style.display = "none";
  }

  // 强制清空 Canvas 以提供视觉反馈（可选，这里先不加闪烁，直接重绘）
  renderComparisons(orig, adv);
}

/**
 * 加载数据URL为 ImageData
 */
function loadImageData(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      try {
        const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
        resolve(id);
      } catch (e) {
        reject(e);
      }
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

/**
 * 渲染三种对比：Raw Diff、Abs Diff、Heatmap
 */
async function renderComparisons(origDataUrl, advDataUrl) {
  try {
    const [orig, adv] = await Promise.all([loadImageData(origDataUrl), loadImageData(advDataUrl)]);
    renderRawDiff(orig, adv, document.getElementById("diffRawCanvas"));
    renderAbsDiff(orig, adv, document.getElementById("diffAbsCanvas"));
    renderHeatmap(orig, adv, document.getElementById("heatmapCanvas"));
    setupOverlay(origDataUrl, advDataUrl);
  } catch (e) {
    console.error(e);
  }
}

/**
 * 绘制签名差值 Raw Diff（加中灰偏移并放大）
 */
function renderRawDiff(orig, adv, canvas) {
  const w = orig.width, h = orig.height;
  canvas.width = w; canvas.height = h;
  const out = new ImageData(w, h);
  const scale = 10;
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    const o = orig.data[idx];
    const a = adv.data[idx];
    let v = 127 + (a - o) * scale;
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    out.data[idx] = v;
    out.data[idx + 1] = v;
    out.data[idx + 2] = v;
    out.data[idx + 3] = 255;
  }
  const ctx = canvas.getContext("2d");
  ctx.putImageData(out, 0, 0);
}

/**
 * 绘制绝对差值 Abs Diff（放大显示）
 */
function renderAbsDiff(orig, adv, canvas) {
  const w = orig.width, h = orig.height;
  canvas.width = w; canvas.height = h;
  const out = new ImageData(w, h);
  const scale = 10;
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    const o = orig.data[idx];
    const a = adv.data[idx];
    let v = Math.abs(a - o) * scale;
    if (v > 255) v = 255;
    out.data[idx] = v;
    out.data[idx + 1] = v;
    out.data[idx + 2] = v;
    out.data[idx + 3] = 255;
  }
  const ctx = canvas.getContext("2d");
  ctx.putImageData(out, 0, 0);
}

/**
 * 绘制扰动热力图 Heatmap（按图内最大差值归一化）
 */
function renderHeatmap(orig, adv, canvas) {
  const w = orig.width, h = orig.height;
  canvas.width = w; canvas.height = h;
  const out = new ImageData(w, h);
  let maxd = 0;
  const diffs = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    const d = Math.abs(adv.data[idx] - orig.data[idx]);
    diffs[i] = d;
    if (d > maxd) maxd = d;
  }
  for (let i = 0; i < w * h; i++) {
    const t = maxd > 0 ? diffs[i] / maxd : 0;
    const [r, g, b] = colormapJet(t);
    const idx = i * 4;
    out.data[idx] = r;
    out.data[idx + 1] = g;
    out.data[idx + 2] = b;
    out.data[idx + 3] = 255;
  }
  const ctx = canvas.getContext("2d");
  ctx.putImageData(out, 0, 0);
}

/**
 * Jet 伪彩色映射
 */
function colormapJet(t) {
  const r = Math.floor(255 * clamp01(1.5 - Math.abs(2 * t - 1.5)));
  const g = Math.floor(255 * clamp01(1.5 - Math.abs(2 * t - 1.0)));
  const b = Math.floor(255 * clamp01(1.5 - Math.abs(2 * t - 0.5)));
  return [r, g, b];
}

/**
 * 叠加对比：绑定透明度滑块
 */
function setupOverlay(origUrl, advUrl) {
  const o = document.getElementById("overlayOrig");
  const a = document.getElementById("overlayAdv");
  o.src = origUrl;
  a.src = advUrl;
  const range = document.getElementById("overlayRange");
  a.style.opacity = range.value;
  range.addEventListener("input", () => {
    a.style.opacity = range.value;
  });
}

/**
 * [0,1] 裁剪
 */
function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}
