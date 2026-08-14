/* 统计子站共享脚本 */
const STATS = LoadGate.load(
  [{ key: "stats", url: "../data/stats.json", type: "json" }],
  { sub: "统计数据(约 0.9 MB)正在下载到你的浏览器,下载完成后即可查看图表。" }
).then((rows) => {
  LoadGate.gate.close();
  return rows[0].value;
}).catch((e) => {
  console.error(e);
  LoadGate.gate.fail("统计数据下载失败,请检查网络后点「重新下载」。");
  return new Promise(() => {}); // 保持挂起,遮罩与重试按钮继续展示
});

const CHART_DEFAULTS = {
  color: "#a7b0d6",
  borderColor: "rgba(255,255,255,.12)",
  font: { family: "'Noto Sans SC', sans-serif", size: 12 },
};

const PALETTE = [
  "#ff6fb5", "#9d7bff", "#5ad8ff", "#ffc96b", "#58e0a1", "#ff8a5c",
  "#7be0ff", "#c9b8ff", "#ff9ad5", "#8be8b8", "#ffb36b", "#8fb6ff",
  "#ff7b7b", "#a5e07b", "#7bd6c8", "#e0a5ff", "#ffd97b", "#7bb8ff",
  "#ffa58c", "#a5c8ff", "#c5ff9a", "#ff9acd", "#9affdc", "#d9a5ff",
];

function mkBar(id, labels, values, opts = {}) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: opts.color || PALETTE[0],
        borderRadius: 6,
        barThickness: opts.thickness || undefined,
      }],
    },
    options: {
      indexAxis: opts.horizontal ? "y" : "x",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: opts.tooltip
            ? { label: opts.tooltip }
            : { label: (c) => `${c.parsed.x ?? c.parsed.y}` },
        },
      },
      scales: {
        x: { grid: { color: "rgba(255,255,255,.06)" }, ticks: { ...CHART_DEFAULTS, autoSkip: true } },
        y: { grid: { color: "rgba(255,255,255,.06)" }, ticks: { ...CHART_DEFAULTS, autoSkip: false } },
      },
      layout: { padding: { left: 6, right: 10 } },
    },
  });
}

const DENS_STOPS = [
  [0.0, [80, 88, 150]],
  [0.35, [157, 123, 255]],
  [0.7, [255, 201, 107]],
  [1.0, [255, 154, 213]],
];
function densityColor(d) {
  const t = Math.max(0, Math.min(1, d || 0));
  for (let i = 1; i < DENS_STOPS.length; i++) {
    if (t <= DENS_STOPS[i][0]) {
      const [t0, c0] = DENS_STOPS[i - 1];
      const [t1, c1] = DENS_STOPS[i];
      const k = (t - t0) / (t1 - t0);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * k);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * k);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * k);
      return `rgb(${r},${g},${b})`;
    }
  }
  const c = DENS_STOPS[DENS_STOPS.length - 1][1];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function mkScatter(id, points, opts = {}) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [{
        data: points,
        backgroundColor: opts.colorFn
          ? (c) => opts.colorFn(c.raw)
          : (opts.pointColor || (() => PALETTE[0])),
        pointRadius: opts.radius || 4,
        pointHoverRadius: 7,
        borderColor: opts.borderColor || "rgba(255,255,255,.35)",
        borderWidth: 0.4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: opts.tooltip || ((c) => c.raw.label || ""),
          },
        },
      },
      scales: {
        x: {
          title: opts.xTitle ? { display: true, text: opts.xTitle, color: "#a7b0d6", font: { size: 12 } } : undefined,
          grid: { color: "rgba(255,255,255,.06)" },
          ticks: CHART_DEFAULTS,
        },
        y: {
          title: opts.yTitle ? { display: true, text: opts.yTitle, color: "#a7b0d6", font: { size: 12 } } : undefined,
          grid: { color: "rgba(255,255,255,.06)" },
          ticks: CHART_DEFAULTS,
        },
      },
    },
  });
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function barList(container, items, valueFn, nameFn, metaFn) {
  const max = Math.max(...items.map(valueFn), 1);
  container.textContent = "";
  for (const it of items) {
    const row = el("div", "list-row");
    row.appendChild(el("span", "lname", nameFn(it)));
    const track = el("span", "lbar-track");
    const bar = el("span", "lbar");
    bar.style.width = `${Math.max(4, (valueFn(it) / max) * 100)}%`;
    track.appendChild(bar);
    row.appendChild(track);
    if (metaFn) {
      const m = el("span", "lmeta", metaFn(it));
      m.title = metaFn(it);
      row.appendChild(m);
    }
    container.appendChild(row);
  }
}
