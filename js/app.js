/* AnimePick v2 — 五档评估 + 自适应轮数 + 品味原型 + 超喜欢共现图 + 分层结果
   Data contract:
     data/model.json    { dims, clusters, quant{lo,span}, prototypes{n,quant{lo,span}},
                          anchors[{id,name_cn}], cluster_tags[[]] }
     data/items.json    [ { id, name, name_cn, date, platform, score, rank,
                            popularity, rating_count, tags[], cluster, cover } ]
     data/embeddings.bin   uint8 [N x dims]
     data/prototypes.bin   uint8 [n_proto x dims]
     data/co_loved.json    { "<itemId>": [[neighborId, jaccard], ...] }
*/
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  /* ---------------- petals ---------------- */
  (function petals() {
    const cv = $("#petals");
    if (!cv || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const ctx = cv.getContext("2d");
    let W, H;
    const resize = () => { W = cv.width = innerWidth; H = cv.height = innerHeight; };
    resize();
    addEventListener("resize", resize);
    const ps = Array.from({ length: 16 }, () => ({
      x: Math.random() * W, y: Math.random() * H, s: 5 + Math.random() * 8,
      vy: 0.4 + Math.random() * 0.8, vx: -0.25 + Math.random() * 0.5,
      a: Math.random() * 6.28, va: 0.005 + Math.random() * 0.02,
      hue: [330, 335, 340, 200][Math.floor(Math.random() * 4)],
    }));
    (function frame() {
      ctx.clearRect(0, 0, W, H);
      for (const p of ps) {
        p.y += p.vy; p.x += p.vx + Math.sin(p.a) * 0.35; p.a += p.va;
        if (p.y > H + 16) { p.y = -16; p.x = Math.random() * W; }
        if (p.x > W + 16) p.x = -16;
        if (p.x < -16) p.x = W + 16;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(Math.sin(p.a) * 0.9);
        ctx.fillStyle = `hsla(${p.hue}, 82%, 80%, .5)`;
        ctx.beginPath();
        ctx.moveTo(0, -p.s * 1.25);
        ctx.bezierCurveTo(p.s * 0.85, -p.s * 0.55, p.s * 0.72, p.s * 0.75, 0, p.s * 0.72);
        ctx.bezierCurveTo(-p.s * 0.72, p.s * 0.75, -p.s * 0.85, -p.s * 0.55, 0, -p.s * 1.25);
        ctx.fill();
        ctx.restore();
      }
      requestAnimationFrame(frame);
    })();
  })();

  /* ---------------- state ---------------- */
  const S = {
    model: null, items: [], byId: new Map(),
    emb: null, proto: null, coLoved: {},
    anchorsPicked: new Set(),
    answers: [],        // {id, act}  act: love|like|mid|dislike|nope
    seen: new Set(),    // 没看过(仍可被推荐)与已问过的条目
    asked: new Set(),   // 所有出现在问答中的条目
    round: 0,
    roundsTarget: 25,
    skipCount: 0,
    askedCount: 0,
    times: [],          // 每题耗时 ms
    qStart: 0,
    finalSeen: new Set(), // 结果页手动标记「看过了」的条目
    MIN_RATING_COUNT: 80,
    MMR_LAMBDA: 0.45,
    HOT_THRESHOLD: 40000,  // 全站热度(popularity,来自 wiki 归档),不是本站爬取人数
  };
  const W = { love: 1.5, like: 1.0, mid: 0.3, dislike: -0.8, nope: -0.4, anchor: 0.4 };
  const titleOf = (it) => it.name_cn || it.name || `#${it.id}`;
  const yearOf = (it) => String(it.date || "").slice(0, 4);

  /* ---------------- model loading ---------------- */
  function decode(buf, quant) {
    const d = S.model.dims;
    const lo = quant.lo, span = quant.span;
    const n = buf.length / d;
    const out = [];
    for (let i = 0; i < n; i++) {
      const v = new Float32Array(d);
      for (let k = 0; k < d; k++) v[k] = lo[k] + span[k] * (buf[i * d + k] / 255);
      out.push(v);
    }
    return out;
  }
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(v) { return Math.sqrt(dot(v, v)) || 1e-8; }

  async function loadModel() {
    const [m, items, embBuf, embPickBuf, protoBuf, coBuf] = await Promise.all([
      fetch("data/model.json").then((r) => r.json()),
      fetch("data/items.json").then((r) => r.json()),
      fetch("data/embeddings.bin").then((r) => r.arrayBuffer()),
      fetch("data/embeddings_picker.bin").then((r) => r.arrayBuffer()),
      fetch("data/prototypes.bin").then((r) => r.arrayBuffer()),
      fetch("data/co_loved.json").then((r) => r.json()),
    ]);
    S.model = m;
    S.items = items;
    S.emb = decode(new Uint8Array(embBuf), m.quant);                    // BPR:排序用
    S.embPick = decode(new Uint8Array(embPickBuf), m.quant_picker || m.quant); // ALS:提问用
    S.proto = decode(new Uint8Array(protoBuf), m.prototypes.quant);
    S.coLoved = coBuf;
    S.maxControversy = Math.max(...items.map((it) => it.controversy || 0), 1);
    for (const it of items) S.byId.set(it.id, it);
    // 同好短评(可缺失:抓取完成后由 finalize 拷入)
    try {
      S.comments = await fetch("data/comments.json").then((r) => r.json());
    } catch {
      S.comments = {};
    }
    // 观众群名称(可选)
    try {
      S.groupNames = (await fetch("data/group_names.json").then((r) => r.json())).names || [];
    } catch {
      S.groupNames = [];
    }
  }

  /* ---------------- profile & scoring ---------------- */
  function profileVector(embSet) {
    const d = S.model.dims;
    const p = new Float32Array(d);
    for (const a of S.answers) {
      const it = S.byId.get(a.id);
      if (!it) continue;
      const v = embSet[S.items.indexOf(it)];
      const w = W[a.act] || 0;
      for (let k = 0; k < d; k++) p[k] += w * v[k];
    }
    for (const id of S.anchorsPicked) {
      const it = S.byId.get(id);
      if (!it) continue;
      const v = embSet[S.items.indexOf(it)];
      for (let k = 0; k < d; k++) p[k] += W.anchor * v[k];
    }
    return p;
  }
  function protoWeightsOf(p) {
    const n = norm(p);
    if (n < 1e-8) return S.proto.map(() => 1 / S.proto.length);
    const sims = S.proto.map((Pk) => dot(Pk, p) / n);
    const mx = Math.max(...sims);
    const ex = sims.map((s) => Math.exp((s - mx) / 0.2));
    const sum = ex.reduce((a, b) => a + b, 0);
    return ex.map((e) => e / sum);
  }
  function blendedParts(p) {
    // 返回 {base, proto, pop} 三个分量数组(排序分 = 0.5*(0.55base+0.35proto+0.05pop) + 0.5*co)
    const n = norm(p);
    const zero = () => S.items.map(() => 0.05 * Math.log10(1));
    if (n < 1e-8) {
      const pop = S.items.map((it) => 0.05 * Math.log10(1 + (it.popularity || 1)));
      return { base: S.items.map(() => 0), proto: S.items.map(() => 0), pop };
    }
    const wts = protoWeightsOf(p);
    const p2 = new Float32Array(S.model.dims);
    for (let k = 0; k < S.proto.length; k++) {
      for (let d = 0; d < p2.length; d++) p2[d] += wts[k] * S.proto[k][d];
    }
    const base = [], proto = [], pop = [];
    for (const it of S.items) {
      const v = S.emb[S.items.indexOf(it)];
      base.push(dot(v, p) / n);
      proto.push(dot(v, p2) / norm(p2));
      pop.push(0.05 * Math.log10(1 + (it.popularity || 1)));
    }
    return { base, proto, pop };
  }
  function blendedScore(p) {
    const n = norm(p);
    if (n < 1e-8) {
      return S.items.map((it) => 0.05 * Math.log10(1 + (it.popularity || 1)));
    }
    const wts = protoWeightsOf(p);
    const p2 = new Float32Array(S.model.dims);
    for (let k = 0; k < S.proto.length; k++) {
      for (let d = 0; d < p2.length; d++) p2[d] += wts[k] * S.proto[k][d];
    }
    return S.items.map((it) => {
      const v = S.emb[S.items.indexOf(it)];
      const base = dot(v, p) / n;
      const proto = dot(v, p2) / norm(p2);
      return 0.55 * base + 0.35 * proto + 0.05 * Math.log10(1 + (it.popularity || 1));
    });
  }
  function coLovedVector(lovedIds) {
    const v = new Float32Array(S.items.length);
    for (const id of lovedIds) {
      const nbrs = S.coLoved[String(id)];
      if (!nbrs) continue;
      for (const [nid, j] of nbrs) {
        const it = S.byId.get(Number(nid));
        if (it) v[S.items.indexOf(it)] += j;
      }
    }
    const mx = Math.max(...v);
    if (mx > 0) for (let i = 0; i < v.length; i++) v[i] /= mx;
    return v;
  }

  /* ---------------- screen 1: anchors ---------------- */
  const coverCache = {};
  function fallbackCover(img, it) {
    const fb = el("div", "cover-fallback", titleOf(it).slice(0, 1));
    fb.title = titleOf(it);
    if (img.parentNode) img.replaceWith(fb);
  }
  function coverFor(img, it) {
    img.decoding = "async";
    img.alt = titleOf(it);
    const fail = () => fallbackCover(img, it);
    if (it.cover) {
      img.src = it.cover;
      img.onerror = fail;
      return;
    }
    if (!(it.id in coverCache)) {
      coverCache[it.id] = fetch(`https://api.bgm.tv/v0/subjects/${it.id}`)
        .then((r) => r.json())
        .then((d) => (d.images && d.images.large) || "")
        .catch(() => "");
    }
    coverCache[it.id].then((url) => {
      if (img.isConnected && url) { img.src = url; img.onerror = fail; }
      else if (img.isConnected) fail();
    });
  }

  function renderAnchors() {
    const grid = $("#anchor-grid");
    grid.textContent = "";
    const anchors = S.model.anchors
      .map((a) => S.byId.get(a.id))
      .filter(Boolean)
      .sort((a, b) => (b.popularity || 0) - (a.popularity || 0));
    const picked = new Set();
    for (const it of anchors) {
      const card = el("div", "anchor-card");
      card.tabIndex = 0;
      card.appendChild(el("span", "check", "✓"));
      const img = el("img");
      img.loading = "lazy";
      coverFor(img, it);
      card.appendChild(img);
      card.appendChild(el("div", "name", titleOf(it)));
      const toggle = () => {
        if (picked.has(it.id)) { picked.delete(it.id); card.classList.remove("on"); }
        else { picked.add(it.id); card.classList.add("on"); }
        S.anchorsPicked = picked;
        $("#anchor-hint").textContent = `已选 ${picked.size} 部`;
        $("#to-quiz").disabled = picked.size < 3;
      };
      card.onclick = toggle;
      card.onkeydown = (e) => { if (e.key === "Enter") toggle(); };
      grid.appendChild(card);
    }
    $("#to-quiz").onclick = () => {
      $("#screen-anchors").hidden = true;
      $("#screen-quiz").hidden = false;
      nextQuestion();
    };
  }

  /* ---------------- screen 2: quiz ---------------- */
  function candidates() {
    const anchorIds = new Set(S.model.anchors.map((a) => a.id));
    return S.items.filter((it) =>
      (it.rating_count || 0) >= S.MIN_RATING_COUNT &&
      !S.asked.has(it.id) &&
      !anchorIds.has(it.id)
    );
  }
  function updateTarget() {
    // 自适应轮数:跳过率高→收短;秒答→拉长。前 3 题不计时(阅读规则期)
    let target = 25;
    if (S.askedCount >= 6) {
      const skipRatio = S.skipCount / S.askedCount;
      if (skipRatio >= 0.4) target = 12;
      else if (skipRatio <= 0.15) target = 40;
      if (S.times.length >= 8) {
        const times = S.times.slice().sort((a, b) => a - b);
        const med = times[Math.floor(times.length / 2)] || 0;
        if (med > 5000) target = Math.min(target, 12);   // 明显犹豫(选项变多,给足思考时间)
        if (med < 2500 && skipRatio <= 0.15) target = 40; // 快速作答=熟悉
      }
    }
    S.roundsTarget = target;
    const tip = $("#quiz-tip");
    if (tip) tip.textContent =
      `目标 ${target} 轮(按你的回答动态调整) · 「没看过」不消耗轮次 · 可随时点「现在给我结果」`;
  }
  function nextQuestion() {
    if (S.round >= S.roundsTarget) { finish(); return; }
    const cands = candidates();
    if (!cands.length) { finish(); return; }
    const p = profileVector(S.embPick);  // 提问画像走 ALS 探索空间
    let pick;
    if (S.answers.length === 0 && S.anchorsPicked.size === 0) {
      let best = -Infinity;
      for (const it of cands) {
        if ((it.popularity || 0) > best) { best = it.popularity || 0; pick = it; }
      }
    } else {
      let bestScore = -Infinity;
      for (const it of cands) {
        const idx = S.items.indexOf(it);
        const rel = dot(S.embPick[idx], p) / norm(p);
        let maxSim = 0;
        for (const id of S.asked) {
          const ai = S.items.indexOf(S.byId.get(id));
          if (ai < 0) continue;
          const sim = dot(S.embPick[ai], S.embPick[idx]);
          if (sim > maxSim) maxSim = sim;
        }
        // 争议度先验:优先问分歧大的条目(能区分人群),热度先验为辅
        const contro = (it.controversy || 0) / S.maxControversy;
        const score = rel - S.MMR_LAMBDA * maxSim + 0.12 * contro
          + 0.05 * Math.log10(1 + (it.popularity || 1));
        if (score > bestScore) { bestScore = score; pick = it; }
      }
    }
    if (!pick) { finish(); return; }
    renderQuestion(pick);
    S.qStart = performance.now();
    S.round++;
    $("#progress-fill").style.width = `${Math.min(100, (S.round / S.roundsTarget) * 100)}%`;
  }
  let currentIt = null;
  function renderQuestion(it) {
    currentIt = it;
    S.asked.add(it.id);
    const stage = $("#quiz-stage");
    stage.textContent = "";
    const card = el("div", "quiz-card");
    const img = el("img");
    coverFor(img, it);
    card.appendChild(img);
    const info = el("div", "quiz-info");
    info.appendChild(el("h2", null, titleOf(it)));
    if (it.name && it.name !== it.name_cn) info.appendChild(el("div", "jp", it.name));
    const meta = el("div", "meta");
    if (yearOf(it)) meta.appendChild(el("span", null, "📅 " + yearOf(it)));
    if (it.platform) meta.appendChild(el("span", null, it.platform));
    if (it.members > 1) meta.appendChild(el("span", null, `系列 ${it.members} 部`));
    if (it.score != null) meta.appendChild(el("span", null, "★ " + it.score));
    info.appendChild(meta);
    card.appendChild(info);
    stage.appendChild(card);
  }
  function bindQuiz() {
    document.querySelectorAll(".quiz-actions .btn").forEach((b) => {
      b.onclick = () => {
        const act = b.dataset.act;
        const it = currentIt;
        if (!it) return;
        S.times.push(performance.now() - S.qStart);
        S.askedCount++;
        if (act === "seen") {           // 没看过:无信号,不耗轮次
          S.skipCount++;
          S.seen.add(it.id);
          S.round--;
          updateTarget();
          nextQuestion();
          return;
        }
        S.answers.push({ id: it.id, act });
        S.seen.add(it.id);
        if (act === "nope") S.skipCount++;
        updateTarget();
        nextQuestion();
      };
    });
    $("#finish-now").onclick = () => finish();
  }

  /* ---------------- screen 3: results ---------------- */
  function finish() {
    $("#screen-quiz").hidden = true;
    $("#screen-results").hidden = false;
    // 兴趣画像(只渲染一次)
    const tagScore = new Map();
    const clusterScore = new Map();
    for (const a of S.answers) {
      const it = S.byId.get(a.id);
      if (!it) continue;
      if (a.act === "love" || a.act === "like") {
        for (const t of (it.tags || []).slice(0, 8)) tagScore.set(t, (tagScore.get(t) || 0) + (a.act === "love" ? 2 : 1));
      }
      clusterScore.set(it.cluster, (clusterScore.get(it.cluster) || 0) + (W[a.act] || 0));
    }
    const topTags = [...tagScore.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
    const topClusters = [...clusterScore.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2);
    const box = $("#profile-box");
    box.textContent = "";
    box.appendChild(el("h3", null, "🎯 你的兴趣画像"));
    const row = el("div", "tag-row");
    const seenTags = new Set();
    // 标签带强度评分(喜欢 +1,超喜欢 +2,取前 8 归一化为 1-10 分)
    const tagMax = Math.max(...topTags.map((x) => x[1]), 1);
    const pushTag = (t, v) => {
      if (!t || seenTags.has(t)) return;
      seenTags.add(t);
      const pill = el("span", "ptag", t);
      if (v !== undefined) {
        const score = Math.max(1, Math.round((v / tagMax) * 10));
        pill.appendChild(el("i", "tscore", String(score)));
      }
      row.appendChild(pill);
    };
    for (const [t, v] of topTags) pushTag(t, v);
    for (const [c] of topClusters) {
      for (const t of (S.model.cluster_tags[c] || []).slice(0, 3)) pushTag(t);
    }
    if (!row.children.length) pushTag("兴趣广泛,再回答几轮会更准");
    box.appendChild(row);
    // 观众群画像:你最接近的品味原型及其心头好
    const wts = protoWeightsOf(profileVector(S.emb));
    const topProto = wts
      .map((w, k) => ({ w, k }))
      .sort((a, b) => b.w - a.w)
      .slice(0, 2);
    for (const tp of topProto) {
      const labels = (S.model.prototypes.labels && S.model.prototypes.labels[tp.k]) || [];
      if (!labels.length) continue;
      const gname = S.groupNames[tp.k] || "";
      const prow = el("div", "proto-row");
      prow.appendChild(el("span", "proto-name",
        gname ? `「${gname}」观众 · ${Math.round(tp.w * 100)}% 像你`
              : `观众群 ${tp.k + 1} · ${Math.round(tp.w * 100)}% 像你`));
      prow.appendChild(el("span", "proto-loves", "心头好:" + labels.join(" / ")));
      box.appendChild(prow);
    }
    renderResults();
  }

  function renderResults() {
    const p = profileVector(S.emb);  // 最终排序画像走 BPR 排序空间
    // 看过 = 锚点勾选 + 任何给出反馈的问答条目 + 结果页手动标记(没看过不算看过)
    const watched = new Set([...S.anchorsPicked, ...S.answers.map((a) => a.id), ...S.finalSeen]);
    const loved = S.answers.filter((a) => a.act === "love").map((a) => a.id);
    const parts = blendedParts(p);
    const co = coLovedVector(loved);
    const scored = S.items
      .filter((it) => !watched.has(it.id))
      .map((it) => {
        const i = S.items.indexOf(it);
        const base = 0.5 * (0.55 * parts.base[i] + 0.35 * parts.proto[i] + parts.pop[i]);
        return { it, i, base, co: 0.5 * co[i], s: base + 0.5 * co[i] };
      });
    scored.sort((a, b) => b.s - a.s);

    // 推荐理由:优先「超喜欢共现」溯源,其次标签契合
    const likedTagSet = new Set();
    for (const a of S.answers.filter((x) => x.act === "love" || x.act === "like")) {
      const x = S.byId.get(a.id);
      if (x) for (const t of (x.tags || []).slice(0, 8)) likedTagSet.add(t);
    }
    const reasonFor = (it) => {
      let bestSrc = null, bestJ = 0;
      for (const x of loved) {
        const nbrs = S.coLoved[String(x)] || [];
        for (const [nid, j] of nbrs) {
          if (Number(nid) === it.id && j > bestJ) { bestJ = j; bestSrc = S.byId.get(x); }
        }
      }
      if (bestSrc) return `💡 深爱《${titleOf(bestSrc)}》的人也深爱这部`;
      const overlap = (it.tags || []).filter((t) => likedTagSet.has(t)).slice(0, 2);
      if (overlap.length) return `💡 与你喜欢的「${overlap.join("、")}」契合`;
      return null;
    };

    const pickGroup = (list, limit) => {
      const out = [];
      const clusterCount = new Map();
      for (const x of list) {
        const c = x.it.cluster;
        if ((clusterCount.get(c) || 0) >= 2) continue;
        out.push(x);
        clusterCount.set(c, (clusterCount.get(c) || 0) + 1);
        if (out.length >= limit) break;
      }
      return out;
    };
    const renderGroup = (gridSel, list) => {
      const grid = $(gridSel);
      grid.textContent = "";
      list.forEach((x, i) => {
        const it = x.it;
        const card = el("div", "rec-card");
        const link = el("a", "rec-link");
        link.href = "https://bgm.tv/subject/" + it.id;
        link.target = "_blank";
        link.rel = "noopener";
        card.appendChild(el("span", "rank-num", String(i + 1)));
        const img = el("img");
        img.loading = "lazy";
        coverFor(img, it);
        link.appendChild(img);
        const body = el("div", "body");
        body.appendChild(el("div", "t", titleOf(it)));
        const meta = el("div", "meta");
        if (yearOf(it)) meta.appendChild(el("span", null, yearOf(it)));
        if (it.members > 1) meta.appendChild(el("span", null, `系列 ${it.members} 部`));
        if (it.score != null) meta.appendChild(el("span", null, "★ " + it.score));
        if ((it.tags || []).length) meta.appendChild(el("span", null, it.tags[0]));
        body.appendChild(meta);
        const reason = reasonFor(it);
        if (reason) body.appendChild(el("div", "rec-reason", reason));
        const cs = (S.comments || {})[String(it.id)];
        if (cs && cs.length) {
          const cq = el("div", "rec-comment");
          cq.appendChild(el("div", "cq", "💬 " + cs[0].text));
          cq.appendChild(el("div", "cu", "— " + cs[0].user + (cs[0].rate ? ` ★${cs[0].rate}` : "") + " 的同好短评"));
          body.appendChild(cq);
        }
        link.appendChild(body);
        card.appendChild(link);
        const seenBtn = el("button", "seen-btn", "✓ 看过了");
        seenBtn.title = "标记为看过,重新推荐没看过的";
        seenBtn.onclick = () => {
          S.finalSeen.add(it.id);
          renderResults();
          const hint = $("#seen-hint");
          hint.hidden = false;
          hint.textContent = `已排除你标记看过的 ${S.finalSeen.size} 部,榜单已更新。`;
        };
        card.appendChild(seenBtn);
        grid.appendChild(card);
      });
    };
    const hot = pickGroup(scored.filter((x) => (x.it.popularity || 0) >= S.HOT_THRESHOLD), 3);
    const niche = pickGroup(scored.filter((x) => (x.it.popularity || 0) < S.HOT_THRESHOLD), 5);
    $("#group-hot").hidden = hot.length === 0;
    $("#group-niche").hidden = niche.length === 0;
    renderGroup("#rec-grid-hot", hot);
    renderGroup("#rec-grid-niche", niche);

    // 算法信号面板:本次推荐各信号的平均贡献占比
    const shown = [...hot, ...niche];
    if (shown.length) {
      let profileSum = 0, protoSum = 0, coSum = 0, popSum = 0;
      for (const x of shown) {
        const i = x.i;
        profileSum += Math.abs(0.5 * 0.55 * parts.base[i]);
        protoSum += Math.abs(0.5 * 0.35 * parts.proto[i]);
        coSum += Math.abs(x.co);
        popSum += Math.abs(0.5 * parts.pop[i]);
      }
      const total = profileSum + protoSum + coSum + popSum || 1;
      const pct = (v) => Math.round((v / total) * 100);
      const sig = $("#signal-panel");
      sig.hidden = false;
      sig.textContent = "";
      sig.appendChild(el("b", null, "🎛 这次推荐的信号构成:"));
      sig.appendChild(el("span", null, `画像匹配 ${pct(profileSum)}%`));
      sig.appendChild(el("span", null, `· 相似观众群 ${pct(protoSum)}%`));
      sig.appendChild(el("span", null, `· 超喜欢共现 ${pct(coSum)}%`));
      sig.appendChild(el("span", null, `· 热度 ${pct(popSum)}%`));
    }
  }

  /* ---------------- Bangumi 连通性体检(首页提示) ---------------- */
  function checkApi() {
    return new Promise((res) => {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 6000);
      fetch("https://api.bgm.tv/v0/subjects/265", { signal: ctrl.signal })
        .then((r) => { clearTimeout(t); res(r.ok); })
        .catch(() => { clearTimeout(t); res(false); });
    });
  }
  function checkImages() {
    return new Promise((res) => {
      const img = new Image();
      const t = setTimeout(() => res(false), 6000);
      img.onload = () => { clearTimeout(t); res(true); };
      img.onerror = () => { clearTimeout(t); res(false); };
      img.src = "https://lain.bgm.tv/pic/cover/l/e5/69/265_Z5Uou.jpg";
    });
  }
  async function healthCheck() {
    // 调试锚点:强制展示「双端不可用」提示(验证 UI)
    if (location.hash === "#api-off") {
      const banner = $("#api-banner");
      $("#api-banner-msg").textContent =
        "⚠️ 无法连接 Bangumi API、无法加载 Bangumi 图床。核心推荐不受影响(数据与模型都在浏览器本地),但番剧封面会显示占位图,「发现」页的在线补图也不可用。";
      banner.hidden = false;
      $("#api-banner-x").onclick = () => { banner.hidden = true; };
      return;
    }
    const [apiOk, imgOk] = await Promise.all([checkApi(), checkImages()]);
    const problems = [];
    if (!apiOk) problems.push("无法连接 Bangumi API");
    if (!imgOk) problems.push("无法加载 Bangumi 图床");
    if (!problems.length) return;
    const banner = $("#api-banner");
    const msg = $("#api-banner-msg");
    let impact;
    if (!apiOk && !imgOk) {
      impact = "核心推荐不受影响(数据与模型都在浏览器本地),但番剧封面会显示占位图,「发现」页的在线补图也不可用。";
    } else if (!apiOk) {
      impact = "核心推荐不受影响(数据与模型都在浏览器本地),但部分封面补取与在线搜索可能不可用。";
    } else {
      impact = "核心推荐不受影响,但番剧封面会显示占位图。";
    }
    msg.textContent = `⚠️ ${problems.join("、")}。${impact}`;
    banner.hidden = false;
    $("#api-banner-x").onclick = () => { banner.hidden = true; };
  }

  /* ---------------- boot ---------------- */
  async function boot() {
    try {
      await loadModel();
    } catch (e) {
      console.error(e);
      document.body.innerHTML =
        '<p style="text-align:center;padding:80px 20px;color:#a7b0d6">模型数据加载失败,请稍后重试。</p>';
      return;
    }
    renderAnchors();
    bindQuiz();
    healthCheck();
    $("#restart").onclick = () => location.reload();
    // 首页开始按钮:进入锚点屏
    $("#start-btn").onclick = () => {
      $("#screen-home").hidden = true;
      $("#screen-anchors").hidden = false;
      scrollTo({ top: 0 });
    };
    // 调试锚点
    if (location.hash === "#quiz" || location.hash === "#demo") {
      $("#screen-home").hidden = true;
      $("#screen-anchors").hidden = false;
      document.querySelectorAll(".anchor-card").forEach((c, i) => { if (i < 3) c.click(); });
      $("#to-quiz").click();
    }
    if (location.hash === "#demo") {
      const acts = ["nope", "mid", "dislike", "like", "love"];
      const iv = setInterval(() => {
        if (!$("#screen-quiz").hidden) {
          const btns = document.querySelectorAll(".quiz-actions .btn");
          if (btns.length) {
            // 30% 概率点「没看过」(模拟真实用户),否则随机给反馈
            if (Math.random() < 0.3) btns[0].click();
            else btns[1 + Math.floor(Math.random() * acts.length)].click();
          }
        } else {
          clearInterval(iv);
          setTimeout(() => {
            const imgs = [...document.images];
            const loaded = imgs.filter((i) => i.naturalWidth > 0).length;
            document.title = `DEMO imgs=${loaded}/${imgs.length}`;
          }, 1500);
        }
      }, 130);
    }
  }

  boot();
})();
