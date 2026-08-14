/* LoadGate — 数据下载门禁
   数据(模型/条目/向量/统计)下载完成前,页面保持全屏遮罩、不可交互,
   遮罩上实时显示下载进度(百分比 + 当前文件 + 已下载/总大小);
   下载失败时给出明确提示与「重新下载」按钮。

   用法:
     1) 页面在业务脚本之前引入本文件(<script src="js/loadgate.js"></script>);
     2) 业务脚本用 LoadGate.load([{key,url,type,optional}]) 下载,
        全部完成后调用 LoadGate.gate.close(),失败调用 LoadGate.gate.fail(msg)。

   说明:
     - 若数据在 ~150ms 内(缓存命中)下载完成,遮罩不闪现;
     - 可选文件(optional:true)404 不影响整体,对应 value 为 undefined;
     - 加载期间页面其余部分被 inert,键盘/点击都无法误操作半初始化状态。
*/
(function () {
  "use strict";

  /* 各数据文件的体积提示(无 Content-Length 时用于估算进度) */
  const SIZE_HINTS = {
    "model.json": 20227,
    "items.json": 658910,
    "embeddings.bin": 99776,
    "embeddings_picker.bin": 99776,
    "prototypes.bin": 4096,
    "co_loved.json": 234004,
    "comments.json": 421065,
    "group_names.json": 1174,
    "stats.json": 880037,
  };
  const basename = (u) => (u.split("?")[0].split("/").pop());
  const kb = (n) => {
    if (!n) return "?";
    return n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : (n / 1024).toFixed(0) + " KB";
  };

  /* ---------------- 遮罩 DOM ---------------- */
  const ov = document.createElement("div");
  ov.className = "loadgate lg-hidden";
  ov.setAttribute("role", "dialog");
  ov.setAttribute("aria-modal", "true");
  ov.setAttribute("aria-label", "正在下载页面数据");
  ov.innerHTML =
    '<div class="lg-box">' +
      '<div class="lg-icon">🎬</div>' +
      '<h2 class="lg-title">正在下载数据…</h2>' +
      '<p class="lg-sub"></p>' +
      '<div class="lg-track"><div class="lg-fill"></div></div>' +
      '<div class="lg-meta">' +
        '<span class="lg-pct">0%</span>' +
        '<span class="lg-file"></span>' +
      "</div>" +
      '<p class="lg-hint">进度长时间不动?数据托管在 GitHub Pages,请检查你的网络连接。</p>' +
      '<button class="btn btn-primary lg-retry" hidden>🔄 重新下载</button>' +
    "</div>";
  document.body.appendChild(ov);
  const q = (s) => ov.querySelector(s);

  let shown = false;
  let showTimer = 0;
  let closed = false;

  function lockPage() {
    for (const n of document.body.children) {
      if (n !== ov) n.inert = true;
    }
    document.documentElement.classList.add("boot-locked");
  }
  function unlockPage() {
    for (const n of document.body.children) {
      if (n !== ov) n.inert = false;
    }
    document.documentElement.classList.remove("boot-locked");
  }

  const gate = {
    show() {
      if (shown || closed) return;
      shown = true;
      ov.classList.remove("lg-hidden");
      lockPage();
    },
    hide() {
      if (!shown) return;
      shown = false;
      unlockPage();
      ov.classList.add("lg-hidden");
    },
    close() {
      if (closed) return;
      closed = true;
      clearTimeout(showTimer);
      unlockPage();
      if (shown) ov.classList.add("lg-done");   // 已显示 → 淡出
      else ov.classList.add("lg-hidden");       // 未显示 → 保持隐藏,不闪烁
      setTimeout(() => ov.remove(), 500);
    },
    fail(msg) {
      clearTimeout(showTimer);
      closed = false;
      this.show();
      ov.classList.add("lg-error");
      q(".lg-icon").textContent = "😿";
      q(".lg-title").textContent = "数据下载失败";
      q(".lg-sub").textContent = msg || "请检查网络后重试。";
      q(".lg-hint").hidden = true;
      q(".lg-file").textContent = "";
      const btn = q(".lg-retry");
      btn.hidden = false;
      btn.onclick = () => location.reload();
    },
    setProgress(pct, fileLabel) {
      const p = Math.max(0, Math.min(100, Math.round(pct)));
      q(".lg-fill").style.width = p + "%";
      q(".lg-pct").textContent = p + "%";
      q(".lg-file").textContent = fileLabel || "";
    },
    setText(sub) {
      q(".lg-sub").textContent = sub || "";
    },
  };

  /* ---------------- 带进度的 fetch ---------------- */
  async function fetchTracked(url, onBytes) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status + " · " + url);
    let total = parseInt(r.headers.get("content-length") || "0", 10);
    // 服务端可能返回 gzip/br:此时 Content-Length 是压缩体积,与流读出的解压字节数不一致,
    // 改用体积提示,保证进度百分比单调正确
    if (!total || r.headers.get("content-encoding")) total = SIZE_HINTS[basename(url)] || 0;
    if (!r.body) {
      const buf = await r.arrayBuffer();
      onBytes(buf.byteLength || total, total, true);
      return buf;
    }
    const reader = r.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onBytes(got, total, false);
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    onBytes(got, total, true);
    return buf.buffer;
  }

  /* ---------------- 批量加载 ---------------- */
  async function load(files, opts) {
    opts = opts || {};
    const state = files.map((f) => ({
      key: f.key, url: f.url, type: f.type || "json",
      optional: !!f.optional, got: 0, total: 0, done: false, value: undefined,
    }));
    const totalBytes = () => {
      let g = 0, t = 0;
      for (const st of state) {
        if (st.done) { g += st.total; t += st.total; }
        else { g += st.got; t += st.total; }
      }
      return [g, t];
    };

    // 150ms 内完成(缓存命中)就不弹遮罩,避免闪烁
    clearTimeout(showTimer);
    showTimer = setTimeout(() => { gate.setText(opts.sub || ""); gate.show(); }, 150);

    const jobs = files.map(async (f, i) => {
      const st = state[i];
      try {
        const buf = await fetchTracked(f.url, (got, total, done) => {
          st.got = got; st.total = total; st.done = done;
          if (!done && total) st.total = Math.max(total, got);
          const [g, t] = totalBytes();
          const pct = t > 0 ? (g / t) * 100 : 0;
          const size = total && !done ? `${kb(got)} / ${kb(total)}` : kb(total || got);
          gate.setProgress(pct, `正在下载 ${basename(f.url)} (${size})`);
        });
        if (st.type === "json") {
          st.value = JSON.parse(new TextDecoder("utf-8").decode(buf));
        } else {
          st.value = buf;
        }
        st.got = st.total = st.total || buf.byteLength;
        st.done = true;
        const [g, t] = totalBytes();
        gate.setProgress(t > 0 ? (g / t) * 100 : 0, "");
      } catch (e) {
        if (st.optional) {
          st.value = undefined;
          st.done = true; // 可选文件失败:跳过,不拖累整体进度
        } else {
          throw e;
        }
      }
    });
    await Promise.all(jobs);
    clearTimeout(showTimer);
    if (shown) gate.setProgress(100, "全部下载完成");
    gate.close();
    return state.map((st) => ({ key: st.key, value: st.value }));
  }

  window.LoadGate = { load, gate };
})();
