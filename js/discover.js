/* AnimePick 发现页:从一部番出发的四维度推荐 */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const titleOf = (it) => it.name_cn || it.name || `#${it.id}`;
  const yearOf = (it) => String(it.date || "").slice(0, 4);

  const S = { items: [], byId: new Map(), emb: [], proto: [], coLoved: {}, model: null };

  function decode(buf, quant) {
    const d = S.model.dims;
    const lo = quant.lo, span = quant.span;
    const out = [];
    for (let i = 0; i < buf.length / d; i++) {
      const v = new Float32Array(d);
      for (let k = 0; k < d; k++) v[k] = lo[k] + span[k] * (buf[i * d + k] / 255);
      out.push(v);
    }
    return out;
  }
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(v) { return Math.sqrt(dot(v, v)) || 1e-8; }

  async function load() {
    const [m, items, embBuf, protoBuf, coBuf] = await Promise.all([
      fetch("data/model.json").then((r) => r.json()),
      fetch("data/items.json").then((r) => r.json()),
      fetch("data/embeddings.bin").then((r) => r.arrayBuffer()),
      fetch("data/prototypes.bin").then((r) => r.arrayBuffer()),
      fetch("data/co_loved.json").then((r) => r.json()),
    ]);
    S.model = m;
    S.items = items;
    S.emb = decode(new Uint8Array(embBuf), m.quant);
    S.proto = decode(new Uint8Array(protoBuf), m.prototypes.quant);
    S.coLoved = coBuf;
    for (const it of items) S.byId.set(it.id, it);
    // 热门速选 chips
    const chips = $("#chips");
    const hot = [...items].sort((a, b) => (b.popularity || 0) - (a.popularity || 0)).slice(0, 12);
    for (const it of hot) {
      const c = el("button", "chip", titleOf(it));
      c.onclick = () => pick(it);
      chips.appendChild(c);
    }
    // 调试锚点:自动选择第一个热门条目
    if (location.hash === "#demo") {
      setTimeout(() => { const c = $("#chips .chip"); if (c) c.click(); }, 600);
    }
  }

  /* ---------------- 搜索 ---------------- */
  const input = $("#q");
  input.addEventListener("input", () => {
    const kw = input.value.trim();
    const box = $("#sr");
    if (!kw) { box.hidden = true; return; }
    const hits = S.items
      .filter((it) => (it.name_cn || "").includes(kw) || (it.name || "").toLowerCase().includes(kw.toLowerCase()))
      .slice(0, 10);
    box.textContent = "";
    box.hidden = false;
    if (!hits.length) {
      box.appendChild(el("div", "sr-empty", "没找到,换个写法试试(支持简体中文/日文)"));
      return;
    }
    for (const it of hits) {
      const row = el("button", "sr-row");
      row.appendChild(el("span", "sr-name", titleOf(it)));
      row.appendChild(el("span", "sr-meta", `${yearOf(it)} · ★${it.score ?? "—"}`));
      row.onclick = () => { box.hidden = true; pick(it); };
      box.appendChild(row);
    }
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const first = $("#sr .sr-row");
      if (first) first.click();
    }
  });

  /* ---------------- 封面 ---------------- */
  const coverCache = {};
  function coverFor(img, it) {
    img.decoding = "async";
    img.alt = titleOf(it);
    if (it.cover) { img.src = it.cover; img.onerror = () => fallback(img, it); return; }
    if (!(it.id in coverCache)) {
      coverCache[it.id] = fetch(`https://api.bgm.tv/v0/subjects/${it.id}`)
        .then((r) => r.json()).then((d) => (d.images && d.images.large) || "").catch(() => "");
    }
    coverCache[it.id].then((url) => {
      if (img.isConnected && url) { img.src = url; img.onerror = () => fallback(img, it); }
      else if (img.isConnected) fallback(img, it);
    });
  }
  function fallback(img, it) {
    const fb = el("div", "cover-fallback", titleOf(it).slice(0, 1));
    if (img.parentNode) img.replaceWith(fb);
  }

  /* ---------------- 推荐计算 ---------------- */
  function neighbors(vSeed, exclude) {
    return S.items
      .filter((it) => !exclude.has(it.id))
      .map((it) => ({ it, s: dot(vSeed, S.emb[S.items.indexOf(it)]) / norm(S.emb[S.items.indexOf(it)]) }))
      .sort((a, b) => b.s - a.s);
  }
  function tagJac(a, b) {
    const ta = new Set((a.tags || []).slice(0, 8));
    const tb = new Set((b.tags || []).slice(0, 8));
    let inter = 0;
    for (const t of ta) if (tb.has(t)) inter++;
    const union = new Set([...ta, ...tb]).size;
    return union ? inter / union : 0;
  }

  function pick(seed) {
    input.value = titleOf(seed);
    const idx = S.items.indexOf(seed);
    const vSeed = S.emb[idx];
    const exclude = new Set([seed.id]);
    const panel = $("#seed-panel");
    panel.hidden = false;
    panel.textContent = "";
    const card = el("div", "seed-card");
    const img = el("img");
    coverFor(img, seed);
    card.appendChild(img);
    const info = el("div", "seed-info");
    info.appendChild(el("h2", null, titleOf(seed)));
    const meta = el("div", "seed-meta");
    if (yearOf(seed)) meta.appendChild(el("span", null, "📅 " + yearOf(seed)));
    if (seed.score != null) meta.appendChild(el("span", null, "★ " + seed.score));
    if (seed.members > 1) meta.appendChild(el("span", null, `系列 ${seed.members} 部`));
    for (const t of (seed.tags || []).slice(0, 5)) meta.appendChild(el("span", null, t));
    info.appendChild(meta);
    card.appendChild(info);
    panel.appendChild(card);

    // 维度 1:口味坐标(64 维余弦)
    const embN = neighbors(vSeed, exclude).slice(0, 3);
    // 维度 2:共同深爱(共现图)
    const coN = ((S.coLoved[String(seed.id)] || []).slice(0, 3)
      .map(([nid, j]) => ({ it: S.byId.get(Number(nid)), s: j }))
      .filter((x) => x.it && !exclude.has(x.it.id)));
    // 维度 3:标签契合
    const tagN = S.items
      .filter((it) => !exclude.has(it.id))
      .map((it) => ({ it, s: tagJac(seed, it) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s).slice(0, 3);
    // 维度 4:同好观众(原型亲和向量相关性)
    const pvSeed = S.proto.map((Pk) => dot(Pk, vSeed) / norm(vSeed));
    const protoN = S.items
      .filter((it) => !exclude.has(it.id))
      .map((it) => {
        const v = S.emb[S.items.indexOf(it)];
        const pv = S.proto.map((Pk) => dot(Pk, v) / norm(v));
        return { it, s: dot(pvSeed, pv) / (norm(pvSeed) * norm(pv) || 1) };
      })
      .sort((a, b) => b.s - a.s).slice(0, 3);

    // 综合:归一化加权
    const parts = [
      [embN, 1.0], [coN, 1.0], [tagN, 0.8], [protoN, 0.8],
    ];
    const agg = new Map();
    for (const [list, w] of parts) {
      const mx = Math.max(...list.map((x) => x.s), 1e-9);
      for (const x of list) {
        const cur = agg.get(x.it.id) || { it: x.it, s: 0 };
        cur.s += (x.s / mx) * w;
        agg.set(x.it.id, cur);
      }
    }
    const overall = [...agg.values()].sort((a, b) => b.s - a.s).slice(0, 3);

    $("#results").hidden = false;
    renderDim("dim-all", "🏆 综合推荐(四维加权)", overall, (x) => `${x.s.toFixed(2)} 综合分`, true);
    renderDim("dim-emb", "🧭 口味坐标相近(64 维向量余弦)", embN, (x) => `相似度 ${x.s.toFixed(3)}`);
    renderDim("dim-co", "💞 共同深爱(给这部打 9-10 分的人也深爱)", coN, (x) => `共现 Jaccard ${x.s.toFixed(3)}`);
    renderDim("dim-tag", "🏷️ 标签契合(标签 Jaccard)", tagN, (x) => `标签重叠 ${(x.s * 100).toFixed(0)}%`);
    renderDim("dim-proto", "👥 同好观众(品味原型亲和相关性)", protoN, (x) => `相关性 ${x.s.toFixed(3)}`);
    scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderDim(id, title, list, metaFn, big) {
    const box = document.getElementById(id);
    box.textContent = "";
    const h = el("h3", "dim-title", title);
    box.appendChild(h);
    if (!list.length) {
      box.appendChild(el("p", "dim-empty", "数据不足:这部番的深爱者太少,暂无可推荐的共同深爱条目。"));
      return;
    }
    const grid = el("div", "dim-cards" + (big ? " big" : ""));
    for (const x of list) {
      const it = x.it;
      const card = el("a", "d-card");
      card.href = "https://bgm.tv/subject/" + it.id;
      card.target = "_blank";
      card.rel = "noopener";
      const img = el("img");
      img.loading = "lazy";
      coverFor(img, it);
      card.appendChild(img);
      const body = el("div", "d-body");
      body.appendChild(el("div", "d-t", titleOf(it)));
      body.appendChild(el("div", "d-m", metaFn(x)));
      card.appendChild(body);
      grid.appendChild(card);
    }
    box.appendChild(grid);
  }

  load();
})();
