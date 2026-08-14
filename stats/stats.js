/* 统计子站共享脚本 */
const STATS = fetch("../data/stats.json").then((r) => r.json());

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

function mkScatter(id, points, opts = {}) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [{
        data: points,
        backgroundColor: opts.pointColor || (() => PALETTE[0]),
        pointRadius: opts.radius || 4,
        pointHoverRadius: 7,
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
