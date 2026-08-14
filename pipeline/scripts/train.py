#!/usr/bin/env python3
"""Train the pilot recommendation model (numpy-only).

Pipeline: ratings -> user/item filtering -> explicit ALS (d dims)
          -> l2-normalized item embeddings -> mini-batch k-means clusters
          -> int8 quantization -> artifacts/ export -> offline cold-start eval
          (Hit@10 / nDCG@10 vs popularity & random baselines)

Usage (from repo root): pipeline/.venv/bin/python pipeline/scripts/train.py [--dims 64] [--clusters 24]
"""
import argparse, json, math, os, random, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")

MIN_USER_RATINGS = 10
MIN_ITEM_RATINGS = 15
# 五档选项的权重映射(与前端一致):超喜欢 9-10,喜欢 7-8,一般 5-6,不喜欢 1-4
WEIGHTS = {10: 1.5, 9: 1.5, 8: 1.0, 7: 1.0, 6: 0.3, 5: 0.3, 4: -0.8, 3: -0.8, 2: -0.8, 1: -0.8}
POS_THRESHOLD = 9  # 评估目标:命中用户「超喜欢」(9-10 分)的条目
N_PROTOTYPES = 64  # 品味原型数量(相似用户聚类)


def w_of(rate):
    return WEIGHTS.get(int(rate), 0.0)


def load_ratings():
    import glob
    rows = []
    seen = {}
    for path in sorted(glob.glob(os.path.join(DATA, "ratings*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue  # tolerate interleaved/corrupted lines from parallel shards
                sid = r.get("subject_id")
                if not sid:
                    continue
                key = (int(r["user"]), int(sid))
                rate = int(r.get("rate") or 0)
                ctype = int(r.get("type") or 0)
                if key in seen:
                    if rate > seen[key][0]:
                        seen[key] = (rate, ctype)
                    continue
                seen[key] = (rate, ctype)
                rows.append((key[0], key[1], rate, ctype))
    print(f"merged {len(rows)} unique rating rows", flush=True)
    return rows


def load_items_meta():
    """Franchise-level metadata: representative = member with max global popularity."""
    from collections import defaultdict as _dd, Counter as _C
    franchises_path = os.path.join(DATA, "franchises.json")
    fmap = {}
    if os.path.exists(franchises_path):
        fmap = {int(k): int(v) for k, v in json.load(open(franchises_path, encoding="utf-8")).items()}
    members = _dd(list)
    meta_raw = {}
    path = os.path.join(DATA, "subjects_filtered.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    sid = int(o["id"])
                except Exception:
                    continue
                fid = fmap.get(sid, sid)
                meta_raw[sid] = o
                members[fid].append(o)
    meta = {}
    for fid, ms in members.items():
        if not ms:
            continue
        rep = max(ms, key=lambda o: o.get("popularity") or 0)
        tagcnt = _C()
        for o in ms:
            for t in (o.get("meta_tags") or []):
                if isinstance(t, str):
                    tagcnt[t] += 1
            for t in (o.get("tags") or []):
                if isinstance(t, dict) and t.get("name"):
                    tagcnt[t["name"]] += 1
                elif isinstance(t, str):
                    tagcnt[t] += 1
        meta[fid] = {
            "id": fid,
            "rep_id": rep["id"],
            "name": rep.get("name") or "",
            "name_cn": rep.get("name_cn") or "",
            "date": rep.get("date") or "",
            "platform": rep.get("platform") or "",
            "score": rep.get("score"),
            "rank": rep.get("rank"),
            "popularity": sum(o.get("popularity") or 0 for o in ms),
            "tags": [t for t, _ in tagcnt.most_common(8)],
            "members": len(ms),
        }
    return meta, fmap


def bpr(R, d, epochs, lr, reg, seed=0):
    """Bayesian Personalized Ranking (SGD): positives = w>0.3 (喜欢/超喜欢),
    negatives = w<0 (不喜欢), users without negatives sample popular unrated items."""
    n, m = R.shape
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.1, (n, d)).astype(np.float32)
    V = rng.normal(0, 0.1, (m, d)).astype(np.float32)
    Rc = R.tocsr()
    pos = []
    neg = []
    for u in range(n):
        row = Rc.getrow(u).toarray().ravel()
        pos.append(np.where(row > 0.3)[0])
        neg.append(np.where(row < 0)[0])
    pop = np.asarray(R.sum(axis=0)).ravel()
    pop_idx = np.argsort(-pop)
    for ep in range(epochs):
        t0 = time.time()
        for u in range(n):
            pi = pos[u]
            if len(pi) == 0:
                continue
            i = pi[rng.integers(len(pi))]
            ni = neg[u]
            if len(ni):
                j = ni[rng.integers(len(ni))]
            else:
                j = pop_idx[rng.integers(0, min(300, m))]
            x = float(U[u] @ (V[i] - V[j]))
            d = 1.0 / (1.0 + np.exp(x))
            Uu = U[u].copy()
            U[u] += lr * (d * (V[i] - V[j]) - reg * U[u])
            V[i] += lr * (d * Uu - reg * V[i])
            V[j] += lr * (-d * Uu - reg * V[j])
        print(f"  bpr epoch {ep + 1}/{epochs} ({time.time() - t0:.1f}s)", flush=True)
    return U, V


def als(R, d, iters, lam):
    """Explicit alternating least squares on sparse rating matrix R (n_users x n_items)."""
    n, m = R.shape
    rng = np.random.default_rng(0)
    U = rng.normal(0, 0.1, (n, d)).astype(np.float32)
    V = rng.normal(0, 0.1, (m, d)).astype(np.float32)
    Rt = R.T.tocsr()
    Rc = R.tocsr()
    for it in range(iters):
        t0 = time.time()
        # solve for users
        for i in range(n):
            idx = Rc.indices[Rc.indptr[i]:Rc.indptr[i + 1]]
            if len(idx) == 0:
                continue
            vals = Rc.data[Rc.indptr[i]:Rc.indptr[i + 1]]
            Vi = V[idx]
            A = Vi.T @ Vi + lam * np.eye(d, dtype=np.float32)
            U[i] = np.linalg.solve(A, Vi.T @ vals)
        # solve for items
        for j in range(m):
            idx = Rt.indices[Rt.indptr[j]:Rt.indptr[j + 1]]
            if len(idx) == 0:
                continue
            vals = Rt.data[Rt.indptr[j]:Rt.indptr[j + 1]]
            Ui = U[idx]
            A = Ui.T @ Ui + lam * np.eye(d, dtype=np.float32)
            V[j] = np.linalg.solve(A, Ui.T @ vals)
        print(f"  als iter {it + 1}/{iters} ({time.time() - t0:.1f}s)", flush=True)
    return U, V


def minibatch_kmeans(X, k, iters=40, batch=1024, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    centers = X[rng.choice(n, k, replace=False)].astype(np.float32).copy()
    counts = np.zeros(k, dtype=np.float32)
    for it in range(iters):
        idx = rng.choice(n, min(batch, n), replace=False)
        batch_x = X[idx]
        # cosine assignment (X is l2-normalized)
        sim = batch_x @ centers.T
        assign = sim.argmax(axis=1)
        for c in range(k):
            sel = batch_x[assign == c]
            if len(sel):
                centers[c] = centers[c] * 0.7 + sel.mean(axis=0) * 0.3
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers /= np.maximum(norms, 1e-8)
    assign = X @ centers.T
    return assign.argmax(axis=1)


def quantize_int8(V):
    """Per-dimension global min/max int8 quantization."""
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    span = hi - lo
    span[span < 1e-6] = 1.0
    q = np.clip(np.round((V - lo) / span * 255.0), 0, 255).astype(np.uint8)
    return q, lo.astype(np.float32).tolist(), span.astype(np.float32).tolist()


def build_profile_matrix(rows, user_map, item_map, n_items):
    """Sparse matrix of profile weights; rate>0 uses w(rate),
    watched-but-unrated (type 3, rate 0) gets a weak implicit positive."""
    import scipy.sparse as sp
    data, r_i, c_i = [], [], []
    for u, s, rate, ctype in rows:
        w = w_of(rate) if rate > 0 else (0.15 if ctype == 3 else 0.0)
        if w == 0.0:
            continue
        if u not in user_map or s not in item_map:
            continue
        data.append(w)
        r_i.append(user_map[u])
        c_i.append(item_map[s])
    return sp.csr_matrix((data, (r_i, c_i)), shape=(len(user_map), n_items), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["als", "bpr"], default="bpr",
                    help="training objective: explicit ALS or BPR ranking loss")
    ap.add_argument("--dims", type=int, default=64)
    ap.add_argument("--clusters", type=int, default=24)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.08)
    ap.add_argument("--beta", type=float, default=0.05,
                    help="popularity prior weight blended into model scoring (log10 popularity)")
    ap.add_argument("--min-item-ratings", type=int, default=MIN_ITEM_RATINGS)
    ap.add_argument("--min-user-ratings", type=int, default=MIN_USER_RATINGS)
    ap.add_argument("--eval-seed-min-rate", type=int, default=5,
                    help="cold-start seeds are sampled from items rated >= this (0 = any rated item)")
    ap.add_argument("--eval-rounds", type=int, default=10,
                    help="active-learning eval rounds")
    args = ap.parse_args()

    rows = load_ratings()
    print(f"loaded {len(rows)} rating rows", flush=True)

    # franchise mapping: collapse series entries; keep max rate per (user, franchise)
    meta, fmap = load_items_meta()
    merged = {}
    for u, s, rate, ctype in rows:
        fid = fmap.get(s, s)
        key = (u, fid)
        if key in merged:
            if rate > merged[key][0]:
                merged[key] = (rate, ctype)
            continue
        merged[key] = (rate, ctype)
    rows = [(u, fid, rate, ctype) for (u, fid), (rate, ctype) in merged.items()]
    print(f"franchise-collapsed: {len(rows)} rows", flush=True)

    # item filtering
    from collections import Counter
    item_count = Counter(s for _, s, _, _ in rows)
    user_count = Counter(u for u, _, _, _ in rows)
    items = sorted(s for s, c in item_count.items() if c >= args.min_item_ratings)
    users = sorted(u for u, c in user_count.items() if c >= args.min_user_ratings)
    print(f"pool: {len(items)} items (>= {args.min_item_ratings} ratings), "
          f"{len(users)} users (>= {args.min_user_ratings})", flush=True)
    if len(items) < 2 or len(users) < 2:
        sys.exit("pool too small for training; crawl more data first")

    item_map = {s: i for i, s in enumerate(items)}
    user_map = {u: i for i, u in enumerate(users)}
    R = build_profile_matrix(rows, user_map, item_map, len(items))
    nnz = R.nnz
    print(f"matrix: {R.shape[0]}x{R.shape[1]}, {nnz} nnz", flush=True)

    if args.model == "bpr":
        U, V = bpr(R, args.dims, epochs=max(20, args.iters * 2), lr=0.05, reg=0.01)
        Ua, Va = als(R, args.dims, max(6, args.iters // 2), args.lam)
        Va = Va / np.maximum(np.linalg.norm(Va, axis=1, keepdims=True), 1e-8)
    else:
        U, V = als(R, args.dims, args.iters, args.lam)
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
    VnA = Va if args.model == "bpr" else Vn

    # controversy: shrunk per-item rating std ("有人狂吹有人狂喷"的条目)
    from collections import defaultdict as _dd2
    rates_by_item = _dd2(list)
    for u, s, rate, _ in rows:
        if rate > 0:
            rates_by_item[s].append(rate)
    stds = [float(np.std(v)) for v in rates_by_item.values() if len(v) >= 3]
    global_std = float(np.mean(stds)) if stds else 1.0
    controversy = {}
    for s, v in rates_by_item.items():
        if len(v) >= 3:
            raw = float(np.std(v))
            controversy[s] = round((len(v) * raw + 8 * global_std) / (len(v) + 8), 3)
        else:
            controversy[s] = round(global_std, 3)

    # taste prototypes: cluster normalized user factors into N representative "user types"
    Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-8)
    proto_assign = minibatch_kmeans(Un, N_PROTOTYPES)
    P = np.zeros((N_PROTOTYPES, args.dims), dtype=np.float32)
    for k in range(N_PROTOTYPES):
        sel = Un[proto_assign == k]
        if len(sel):
            P[k] = sel.mean(axis=0)
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-8)
    print(f"taste prototypes: {N_PROTOTYPES} from {U.shape[0]} users", flush=True)

    # 每个品味原型的「心头好」标签:该原型偏好最强的热门条目名
    prototype_labels = []
    well_known = [j for j, sid in enumerate(items) if item_count[sid] >= 150]
    for k in range(N_PROTOTYPES):
        if well_known:
            jj = sorted(well_known, key=lambda j: -float(P[k] @ Vn[j]))[:3]
            prototype_labels.append([
                (meta.get(items[j], {}).get("name_cn") or meta.get(items[j], {}).get("name") or f"#{items[j]}")
                for j in jj
            ])
        else:
            prototype_labels.append([])

    print("clustering...", flush=True)
    assign = minibatch_kmeans(Vn, args.clusters)

    # quantization of normalized item vectors
    q, lo, span = quantize_int8(Vn)
    qA, loA, spanA = quantize_int8(VnA)
    pq, plo, pspan = quantize_int8(P)

    # metadata merge (franchise-level)
    items_out = []
    for j, sid in enumerate(items):
        m = meta.get(sid, {})
        items_out.append({
            "id": sid,
            "rep_id": m.get("rep_id") or sid,
            "name": m.get("name") or f"subject {sid}",
            "name_cn": m.get("name_cn") or "",
            "date": m.get("date") or "",
            "platform": m.get("platform") or "",
            "score": m.get("score"),
            "rank": m.get("rank"),
            "popularity": m.get("popularity") or item_count[sid],
            "rating_count": item_count[sid],
            "members": m.get("members") or 1,
            "tags": (m.get("tags") or [])[:8],
            "controversy": controversy.get(sid, global_std),
            "cluster": int(assign[j]),
        })
    # anchors: per-cluster best candidate (score x popularity), then franchise
    # dedup + score floor + fill to a target count
    from collections import Counter as _C
    import re as _re

    def franchise_key(name_cn):
        s = name_cn or ""
        s = _re.sub(r"[ ～].*$", "", s)
        s = _re.sub(r"(第[一二三四五六七八九十0-9]+季|第二季|第三季|第四季|第五季|S[0-9]+).*$", "", s)
        return s[:8] or s

    ANCHOR_TARGET = 20
    ANCHOR_MIN_SCORE = 6.5
    cluster_cands = []
    for c in range(args.clusters):
        cands = [it for it in items_out if it["cluster"] == c and it["rating_count"] >= 150]
        if not cands:
            cands = [it for it in items_out if it["cluster"] == c]
        cands.sort(key=lambda x: -((x["score"] or 0) * min(x["rating_count"], 1000)))
        cluster_cands.append(cands)
    # first pass: best valid candidate per cluster (score floor + franchise dedup)
    anchors, used_franchise, used_clusters = [], set(), set()
    for c, cands in enumerate(cluster_cands):
        for it in cands:
            if (it["score"] or 0) < ANCHOR_MIN_SCORE:
                continue
            fk = franchise_key(it["name_cn"] or it["name"])
            if fk in used_franchise:
                continue
            anchors.append({"id": it["id"], "name_cn": it["name_cn"] or it["name"]})
            used_franchise.add(fk)
            used_clusters.add(c)
            break
    # second pass: fill missing clusters with their best remaining candidate
    for c, cands in enumerate(cluster_cands):
        if c in used_clusters:
            continue
        for it in cands:
            if (it["score"] or 0) < ANCHOR_MIN_SCORE:
                continue
            fk = franchise_key(it["name_cn"] or it["name"])
            if fk in used_franchise:
                continue
            anchors.append({"id": it["id"], "name_cn": it["name_cn"] or it["name"]})
            used_franchise.add(fk)
            used_clusters.add(c)
            break
    anchors.sort(key=lambda a: -item_count[a["id"]])
    if len(anchors) > ANCHOR_TARGET:
        anchors = anchors[:ANCHOR_TARGET]
    # top tags per cluster (for explanations)
    cluster_tags = []
    for c in range(args.clusters):
        cnt = _C()
        for it in items_out:
            if it["cluster"] == c:
                for t in it["tags"][:8]:
                    cnt[t] += 1
        cluster_tags.append([t for t, _ in cnt.most_common(5)])
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items_out, f, ensure_ascii=False)
    q.tofile(os.path.join(ART, "embeddings.bin"))
    qA.tofile(os.path.join(ART, "embeddings_picker.bin"))
    pq.tofile(os.path.join(ART, "prototypes.bin"))
    with open(os.path.join(ART, "model.json"), "w", encoding="utf-8") as f:
        json.dump({"dims": args.dims, "clusters": args.clusters,
                   "quant": {"lo": lo, "span": span},
                   "quant_picker": {"lo": loA, "span": spanA},
                   "prototypes": {"n": N_PROTOTYPES,
                                  "quant": {"lo": plo, "span": pspan},
                                  "labels": prototype_labels},
                   "anchors": anchors, "cluster_tags": cluster_tags}, f)
    print(f"exported artifacts: {len(items_out)} items, dims={args.dims}, "
          f"embeddings.bin={q.nbytes} bytes", flush=True)

    # ---------------- co-loved graph (9-10 分共现) ----------------
    # "给 X 打 9-10 分的人还深爱哪些动画" — 直接命中「找特别喜欢的类型」
    from collections import defaultdict as _dd
    supers = _dd(set)
    for u, s, r, _ in rows:
        if r >= 9 and s in item_map:
            supers[u].add(s)
    co = Counter()
    n9 = Counter()
    for u, ss in supers.items():
        lst = sorted(ss)
        for s in lst:
            n9[s] += 1
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                co[(lst[a], lst[b])] += 1
                co[(lst[b], lst[a])] += 1
    graph = _dd(list)
    for (a, b), c in co.items():
        if c >= 2 and a != b:
            j = c / (n9[a] + n9[b] - c)
            graph[a].append((b, round(j, 4)))
    for a in graph:
        graph[a] = sorted(graph[a], key=lambda x: -x[1])[:20]
    with open(os.path.join(ART, "co_loved.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in graph.items()}, f)
    print(f"co-loved graph: {len(graph)} items with super-like neighbors "
          f"(users with 9-10 ratings: {len(supers)})", flush=True)

    # ---------------- stats export (统计页数据) ----------------
    item_by_id = {it["id"]: it for it in items_out}
    # 1) 评分分布(系列级)
    rate_counts = Counter(r for _, _, r, _ in rows if r > 0)
    rating_dist = {str(k): int(rate_counts.get(k, 0)) for k in range(1, 11)}
    # 2) 全站 TOP 标签
    tag_counts = Counter()
    for it in items_out:
        for t in it["tags"]:
            tag_counts[t] += 1
    top_tags = [[t, int(c)] for t, c in tag_counts.most_common(20)]
    # 3) 意外动画对:用户重叠高(共现 Jaccard)但标签重叠低
    n9 = Counter(s for u, s, r, _ in rows if r >= 9)
    pair_seen = set()
    pair_cloud = []   # 全量候选对(背景灰点):所有共现图边
    pairs = []        # 高亮对:意外度 TOP24 且共同深爱 ≥4
    for a, nbrs in graph.items():
        ia = item_by_id.get(a)
        if not ia:
            continue
        for b, j in nbrs:
            ib = item_by_id.get(b)
            if not ib:
                continue
            key = tuple(sorted((a, b)))
            if key in pair_seen:
                continue
            pair_seen.add(key)
            ta = set(ia["tags"][:8])
            tb = set(ib["tags"][:8])
            shared = sorted(ta & tb)
            tag_j = len(shared) / max(1, min(len(ta), len(tb)))
            shared_n = int(round(j * (n9[a] + n9[b]) / (1 + j)))
            surprise = j * (1.0 - tag_j)
            pair_cloud.append([round(j, 4), round(tag_j, 4)])
            # 自适应阈值:共同深爱 ≥ max(6, 6% × 两部中较小的超喜欢人数)
            thr = max(6, int(0.06 * min(n9[a], n9[b])))
            if shared_n < thr:  # 过滤小样本噪声对
                continue
            name_a = ia["name_cn"] or ia["name"]
            name_b = ib["name_cn"] or ib["name"]
            if not name_a or not name_b:
                continue
            pairs.append({
                "a": {"id": a, "name": name_a,
                      "tags": ia["tags"][:6], "pop": ia["popularity"]},
                "b": {"id": b, "name": name_b,
                      "tags": ib["tags"][:6], "pop": ib["popularity"]},
                "userOverlap": round(j, 4),
                "tagOverlap": round(tag_j, 4),
                "sharedTags": shared[:6],
                "sharedLovers": shared_n,
                "surprise": round(surprise, 4),
            })
    pairs.sort(key=lambda x: -x["surprise"])
    pairs = pairs[:24]
    # 观众群自动命名:用「差异标签」里 lift 最高的内容性标签组合,不足时用心头好锚点
    NO_NAME_TAGS = {"日本", "TV", "剧场版", "OVA", "WEB", "短片", "动画漫画"}

    def group_name(k, diff, avoid, loves):
        import re as _re2
        cands = [d["tag"] for d in diff if d["tag"] not in NO_NAME_TAGS]
        anchor = ""
        if loves:
            anchor = _re2.sub(r"[!！～~+ ]+.*$", "", loves[0])[:6]
        if len(cands) >= 2:
            return ("·".join(cands[:2]))[:12]
        if cands:
            return (cands[0] + (f"·{anchor}" if anchor else ""))[:12]
        return f"{anchor}同好"[:12] if anchor else f"观众群 {k + 1}"

    # 4) 64 类观众群画像:规模、心头好、差异标签(lift)
    total_users = U.shape[0]
    group_stats = []
    liked_item_idx = {}
    Rc2 = R.tocsr()
    for u in range(total_users):
        idx = Rc2.indices[Rc2.indptr[u]:Rc2.indptr[u + 1]]
        vals = Rc2.data[Rc2.indptr[u]:Rc2.indptr[u + 1]]
        liked_item_idx[u] = [int(i) for i, w in zip(idx, vals) if w > 0.3]
    for k in range(N_PROTOTYPES):
        members = [u for u in range(total_users) if proto_assign[u] == k]
        size = len(members)
        # 组内喜欢条目的标签分布(条目级,去重),lift 与全局条目标签频率对比
        group_items = set()
        for u in members:
            group_items.update(liked_item_idx.get(u, []))
        gtags = Counter()
        for i in group_items:
            for t in items_out[i]["tags"][:8]:
                gtags[t] += 1
        n_group_items = max(1, len(group_items))
        diff = []
        avoid = []
        for t, c in gtags.most_common(60):
            if t in ("日本", "TV"):
                continue
            global_freq = tag_counts[t] / len(items_out)
            group_freq = c / n_group_items
            lift = group_freq / max(global_freq, 1e-6)
            if c >= 3 and lift >= 1.1:
                diff.append({"tag": t, "lift": round(lift, 2)})
            elif c >= 2 and lift <= 0.75:
                avoid.append({"tag": t, "lift": round(lift, 2)})
        diff = diff[:6]
        diff.sort(key=lambda x: -x["lift"])
        avoid.sort(key=lambda x: x["lift"])
        avoid = avoid[:5]
        group_stats.append({
            "k": int(k),
            "size": size,
            "name": group_name(k, diff, avoid, prototype_labels[k] if k < len(prototype_labels) else []),
            "loves": prototype_labels[k] if k < len(prototype_labels) else [],
            "diffTags": diff,
            "avoidTags": avoid,
        })
    # 5) 系列规模 TOP
    franchise_sizes = sorted(
        [{"name": (m.get("name_cn") or m.get("name") or f"#{fid}")[:24],
          "members": m.get("members") or 1}
         for fid, m in meta.items()],
        key=lambda x: -x["members"])[:12]

    # 6) 口碑与争议(系列级评分统计)
    from collections import defaultdict as _dd3
    PLATFORM_NAMES = {1: "TV", 2: "OVA", 3: "剧场版", 4: "短片", 5: "WEB", 2006: "动画漫画"}
    fstats = _dd3(lambda: {"n": 0, "sum": 0.0, "n9": 0, "nlow": 0, "sq": 0.0})
    for u, s, r, _ in rows:
        if r <= 0:
            continue
        st = fstats[s]
        st["n"] += 1; st["sum"] += r; st["sq"] += r * r
        if r >= 9:
            st["n9"] += 1
        if r <= 4:
            st["nlow"] += 1
    rep = []
    for sid, st in fstats.items():
        if st["n"] < 15:
            continue
        m = meta.get(sid, {})
        mean = st["sum"] / st["n"]
        var = max(st["sq"] / st["n"] - mean * mean, 0.0)
        rep.append({
            "id": sid,
            "name": (m.get("name_cn") or m.get("name") or f"#{sid}")[:24],
            "n": st["n"],
            "n9": st["n9"],
            "mean": round(mean, 2),
            "std": round(var ** 0.5, 2),
            "superRate": round(st["n9"] / st["n"], 3),
            "hateRate": round(st["nlow"] / st["n"], 3),
            "bimodality": round(2 * min(st["n9"], st["nlow"]) / st["n"], 3),
            "pop": m.get("popularity") or 0,
            "year": (m.get("date") or "????")[:4],
            "platform": PLATFORM_NAMES.get(m.get("platform"), m.get("platform") or "其他"),
        })
    # 爱恨分明榜 & 好评不动心 & 遗珠
    import bisect as _bisect
    pops_sorted = sorted(x["pop"] for x in rep)
    # 同侪组:均分 ≥7.6 且评分人数 ≥50 的高分动画(「好评不动心」的参照系)
    peer = [x for x in rep if x["n"] >= 50 and x["mean"] >= 7.6]
    peer_sr = sorted(x["superRate"] for x in peer)
    for x in rep:
        # 热度百分位(全站 ≥15 评分):「前 X% 低」= 热度不高于它的比例
        x["popPct"] = round(_bisect.bisect_right(pops_sorted, x["pop"]) / max(1, len(rep)) * 100, 1)
        # 超喜欢率百分位(同侪组内):「前 X% 低」= 超喜欢率不高于它的比例
        x["superPct"] = round(_bisect.bisect_right(peer_sr, x["superRate"]) / max(1, len(peer)) * 100, 1)
    love_hate = sorted([x for x in rep if x["n"] >= 30],
                       key=lambda x: -x["bimodality"])[:15]
    n_rep = len(rep)
    rep_sorted_mean = sorted(rep, key=lambda x: -x["mean"])
    # 「好评但不动心」:均值高但超喜欢率低;「遗珠」:超喜欢率高但热度低
    dull = sorted([x for x in rep if x["n"] >= 50 and x["mean"] >= 7.6],
                  key=lambda x: x["superRate"])[:12]
    hidden_gem = sorted([x for x in rep if x["n"] >= 20 and x["superRate"] >= 0.22],
                        key=lambda x: x["pop"])[:12]

    # 7) 年代偏见(按十年聚合,加权)
    decade_agg = _dd3(lambda: {"n": 0, "sum": 0.0, "n9": 0, "ssq": 0.0})
    for x in rep:
        y = x["year"]
        if not y.isdigit():
            continue
        d = f"{int(y) // 10 * 10}s"
        decade_agg[d]["n"] += x["n"]; decade_agg[d]["sum"] += x["mean"] * x["n"]
        decade_agg[d]["n9"] += x["n9"]; decade_agg[d]["ssq"] += (x["std"] ** 2) * x["n"]
    eras = []
    for d in sorted(decade_agg):
        a = decade_agg[d]
        eras.append({"decade": d, "n": a["n"],
                     "mean": round(a["sum"] / a["n"], 2),
                     "superRate": round(a["n9"] / a["n"], 3),
                     "controversy": round((a["ssq"] / a["n"]) ** 0.5, 2)})
    # 8) 载体差异
    plat_agg = _dd3(lambda: {"n": 0, "sum": 0.0, "n9": 0})
    for x in rep:
        p = str(x["platform"]) or "其他"
        plat_agg[p]["n"] += x["n"]; plat_agg[p]["sum"] += x["mean"] * x["n"]
        plat_agg[p]["n9"] += x["n9"]
    platforms = [{"platform": p, "n": a["n"],
                  "mean": round(a["sum"] / a["n"], 2),
                  "superRate": round(a["n9"] / a["n"], 3)}
                 for p, a in sorted(plat_agg.items(), key=lambda kv: -kv[1]["n"])]

    # 9) 入坑点:系列内收藏数最多的成员条目(原始条目级数据)
    import glob as _g
    member_count = Counter()
    subject_name = {}
    pool_path2 = os.path.join(DATA, "subjects_filtered.jsonl")
    if os.path.exists(pool_path2):
        with open(pool_path2, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    subject_name[int(o["id"])] = o.get("name_cn") or o.get("name") or ""
                except Exception:
                    pass
    for path in _g.glob(os.path.join(DATA, "ratings*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    member_count[int(r["subject_id"])] += 1
                except Exception:
                    pass
    best_entry = {}
    franchise_total = {}
    for sid, c in member_count.items():
        fid = fmap.get(sid, sid)
        if fid not in best_entry or c > best_entry[fid][1]:
            best_entry[fid] = (sid, c)
        franchise_total[fid] = franchise_total.get(fid, 0) + c
    entries = []
    for x in rep:
        fid = x["id"]
        if (meta.get(fid, {}).get("members") or 1) < 5:  # 只看长系列(≥5 部),入坑选择才有意义
            continue
        fname = meta.get(fid, {}).get("name_cn") or ""
        if not fname:
            continue
        if fid in best_entry:
            sid, c = best_entry[fid]
            total = franchise_total.get(fid, 0) or 1
            entries.append({
                "franchise": fname[:18],
                "members": meta.get(fid, {}).get("members") or 0,
                "entry": (subject_name.get(sid) or f"#{sid}")[:18],
                "count": c,
                "share": round(c / total * 100, 1),
            })
    entries.sort(key=lambda x: -x["count"])
    # 10) 用户评分人格(原始条目级)
    user_rates = _dd3(list)
    for path in _g.glob(os.path.join(DATA, "ratings*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("rate"):
                        user_rates[int(r["user"])].append(int(r["rate"]))
                except Exception:
                    pass
    personalities = []
    for u, rs in user_rates.items():
        if len(rs) < 8:
            continue
        mean = np.mean(rs)
        personalities.append({"user": u, "n": len(rs),
                              "mean": round(float(mean), 2),
                              "std": round(float(np.std(rs)), 2)})
    # 11) 口味枢纽:共现图真实度数(截断前的原始计数)
    hub_deg = Counter()
    for (a, b), c in co.items():
        hub_deg[a] += 1
    hubs = sorted([{"id": a, "name": (item_by_id.get(a, {}).get("name_cn") or "")[:22],
                    "degree": d}
                   for a, d in hub_deg.items() if d >= 5], key=lambda x: -x["degree"])[:15]
    # 12) 反向意外对:标签重叠高但观众重叠低(TOP150 热门系列两两)
    top_ids = sorted([x["id"] for x in rep], key=lambda i: -fstats[i]["n"])[:150]
    super_sets = _dd3(set)
    for u, s, r, _ in rows:
        if r >= 9:
            super_sets[u].add(s)
    item_super = _dd3(set)
    for u, ss in super_sets.items():
        for s in ss:
            item_super[s].add(u)
    reverse_pairs = []
    for ai in range(len(top_ids)):
        a = top_ids[ai]
        sa = item_super.get(a)
        ta = set(item_by_id.get(a, {}).get("tags", [])[:8])
        for bi in range(ai + 1, len(top_ids)):
            b = top_ids[bi]
            sb = item_super.get(b)
            if not sa or not sb:
                continue
            inter = len(sa & sb)
            union = len(sa | sb)
            if inter < 2 or union < 10:
                continue
            user_j = inter / union
            tb = set(item_by_id.get(b, {}).get("tags", [])[:8])
            tag_j = len(ta & tb) / max(1, min(len(ta), len(tb)))
            if tag_j >= 0.5 and user_j <= 0.08:
                na = item_by_id.get(a, {}).get("name_cn") or ""
                nb = item_by_id.get(b, {}).get("name_cn") or ""
                if not na or not nb:
                    continue
                reverse_pairs.append({
                    "a": na[:20],
                    "b": nb[:20],
                    "tagOverlap": round(tag_j, 3), "userOverlap": round(user_j, 3),
                    "sharedTags": sorted(ta & tb)[:6],
                    "sharedLovers": inter,
                })
    reverse_pairs.sort(key=lambda x: (x["userOverlap"], -x["tagOverlap"]))
    reverse_pairs = reverse_pairs[:15]

    stats = {
        "ratingDist": rating_dist,
        "topTags": top_tags,
        "pairs": pairs,
        "pairCloud": pair_cloud,
        "reversePairs": reverse_pairs,
        "groups": group_stats,
        "franchiseSizes": franchise_sizes,
        "reputation": rep,
        "loveHate": love_hate,
        "dull": dull,
        "hiddenGem": hidden_gem,
        "eras": eras,
        "platforms": platforms,
        "entries": entries[:15],
        "personalities": personalities,
        "hubs": hubs,
        "pool": {"items": len(items_out), "users": len(users),
                 "rows": len(rows)},
    }
    # 13) 口味地图 2D 投影:t-SNE 合并投影(条目嵌入 + 原型,同一坐标系)
    try:
        from sklearn.manifold import TSNE
        M = np.vstack([Vn, P])
        emb2 = TSNE(n_components=2, perplexity=min(30, max(5, (M.shape[0] - 1) // 3)),
                    init="pca", random_state=0, learning_rate="auto",
                    max_iter=1000).fit_transform(M)
        item_xy = emb2[: len(items_out)]
        proto_xy = emb2[len(items_out):]
        proj = "tsne"
    except Exception:
        # 回退到 SVD(同坐标系)
        def svd2(M2):
            Mc = M2 - M2.mean(axis=0)
            u, s, vt = np.linalg.svd(Mc, full_matrices=False)
            return (Mc @ vt[:2].T).astype(np.float32)
        Mc = np.vstack([Vn, P])
        emb2 = svd2(Mc)
        item_xy = emb2[: len(items_out)]
        proto_xy = emb2[len(items_out):]
        proj = "svd"
    stats["map"] = {
        "proj": proj,
        "items": [{"id": it["id"], "name": (it["name_cn"] or it["name"] or f"#{it['id']}")[:24],
                   "x": round(float(item_xy[j][0]), 3), "y": round(float(item_xy[j][1]), 3),
                   "cluster": it["cluster"], "pop": it["popularity"],
                   "proto": int(np.argmax(P @ Vn[j]))}
                  for j, it in enumerate(items_out)],
        "protos": [{"k": k, "x": round(float(proto_xy[k][0]), 3),
                    "y": round(float(proto_xy[k][1]), 3),
                    "size": group_stats[k]["size"] if k < len(group_stats) else 0,
                    "name": group_stats[k]["name"] if k < len(group_stats) else f"观众群 {k + 1}",
                    "loves": prototype_labels[k] if k < len(prototype_labels) else []}
                   for k in range(N_PROTOTYPES)],
    }
    with open(os.path.join(ART, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)
    with open(os.path.join(ART, "group_names.json"), "w", encoding="utf-8") as f:
        json.dump({"names": [g["name"] for g in group_stats]}, f, ensure_ascii=False)
    print(f"stats.json written: {len(pairs)} pairs, {len(group_stats)} groups", flush=True)

    # ---------------- offline cold-start eval ----------------
    print("evaluating...", flush=True)
    rng = random.Random(42)
    holdout = set(rng.sample(users, max(1, len(users) // 5)))
    train_rows = [r for r in rows if r[0] not in holdout and r[0] in user_map and r[1] in item_map]
    train_users = [u for u in users if u not in holdout]
    train_map = {u: i for i, u in enumerate(train_users)}
    Rt = build_profile_matrix(train_rows, train_map, item_map, len(items))
    if args.model == "bpr":
        Ut, Vt = bpr(Rt, args.dims, epochs=20, lr=0.05, reg=0.01, seed=7)
        # 组合模式:提问(探索)用 ALS 嵌入,最终排序用 BPR 嵌入
        UtA, VtA = als(Rt, args.dims, max(4, args.iters // 2), args.lam)
        VtA = VtA / np.maximum(np.linalg.norm(VtA, axis=1, keepdims=True), 1e-8)
    else:
        Ut, Vt = als(Rt, args.dims, max(4, args.iters // 2), args.lam)
    Vtn = Vt / np.maximum(np.linalg.norm(Vt, axis=1, keepdims=True), 1e-8)
    # training-side taste prototypes (no holdout leakage)
    Utn = Ut / np.maximum(np.linalg.norm(Ut, axis=1, keepdims=True), 1e-8)
    pa = minibatch_kmeans(Utn, N_PROTOTYPES)
    Pt = np.zeros((N_PROTOTYPES, args.dims), dtype=np.float32)
    for k in range(N_PROTOTYPES):
        sel = Utn[pa == k]
        if len(sel):
            Pt[k] = sel.mean(axis=0)
    Pt /= np.maximum(np.linalg.norm(Pt, axis=1, keepdims=True), 1e-8)

    def blended(p):
        """Frontend-matching score: profile 55% + taste-prototype 35% + popularity prior."""
        n = np.linalg.norm(p)
        if n < 1e-8:
            return args.beta * np.log10(pop_scores.astype(np.float32) + 1)
        base = (Vtn @ p) / n
        sims = Pt @ p / n  # p is roughly unit-consistent
        ex = np.exp((sims - sims.max()) / 0.2)
        wts = ex / ex.sum()
        p2 = wts @ Pt
        proto = (Vtn @ p2) / (np.linalg.norm(p2) or 1e-8)
        s = 0.55 * base + 0.35 * proto + args.beta * np.log10(pop_scores.astype(np.float32) + 1)
        return s

    def score_profile(rated):
        p = np.zeros(args.dims, dtype=np.float32)
        for sid, rate in rated:
            j = item_map.get(sid)
            if j is not None and w_of(rate) != 0:
                p += w_of(rate) * Vtn[j]
        if np.linalg.norm(p) < 1e-8:
            return None
        return blended(p)

    import scipy.sparse as sp
    pop_scores = np.array([item_count[s] for s in items], dtype=np.float32)

    def split_seed(rated):
        """Split a user's rated items: up to 5 random 'seed' items, rest are targets.
        Seeds come from items rated >= eval_seed_min_rate (mirrors users picking
        anchors among anime they watched/liked)."""
        pool = [x for x in rated if int(x[1]) >= args.eval_seed_min_rate]
        if not pool:
            pool = rated
        if len(pool) <= 5:
            seed = pool
        else:
            idxs = rng.sample(range(len(pool)), 5)
            seed = [pool[i] for i in idxs]
        seed_ids = {s for s, _ in seed}
        rest = [x for x in rated if x[0] not in seed_ids]
        return seed, rest

    def eval_user(rated):
        seed, rest = split_seed(rated)
        seed_ids = {s for s, _ in seed}
        positives = [s for s, r in rest if int(r) >= POS_THRESHOLD and s in item_map]
        if len(positives) < 1:
            return None
        pv = score_profile(seed)
        if pv is None:
            return None
        s = pv.copy()
        mask = np.ones(len(items), dtype=bool)
        for sid in seed_ids:
            j = item_map.get(sid)
            if j is not None:
                mask[j] = False
        s[~mask] = -1e9
        order = np.argsort(-s)
        hits = 0
        dcg = 0.0
        k = 10
        pos_set = set(positives)
        for rank, j in enumerate(order[:k]):
            sid = items[j]
            if sid in pos_set:
                hits += 1
                dcg += 1.0 / math.log2(rank + 2)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(pos_set))))
        return hits, dcg / max(idcg, 1e-9), 1

    def run_eval(name, scorer):
        hits = dcg = users_done = 0
        for u in holdout:
            rated = [(s, r) for (uu, s, r, _) in rows if uu == u]
            if len(rated) < 4:
                continue
            if name == "model":
                r = eval_user(rated)
                if r is None:
                    continue
                hits += r[0]
                dcg += r[1]
                users_done += 1
                continue
            seed, rest = split_seed(rated)
            seed_ids = {s for s, _ in seed}
            positives = [s for s, r in rest if int(r) >= POS_THRESHOLD and s in item_map]
            if len(positives) < 1:
                continue
            mask = np.ones(len(items), dtype=bool)
            for sid in seed_ids:
                j = item_map.get(sid)
                if j is not None:
                    mask[j] = False
            s = scorer()
            s[~mask] = -1e9
            order = np.argsort(-s)
            pos_set = set(positives)
            h = sum(1 for j in order[:10] if items[j] in pos_set)
            d = sum(1.0 / math.log2(i + 2) for i, j in enumerate(order[:10]) if items[j] in pos_set)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(pos_set))))
            hits += h
            dcg += d / max(idcg, 1e-9)
            users_done += 1
        return hits / max(users_done * 10, 1), dcg / max(users_done, 1)

    h_m, n_m = run_eval("model", None)
    h_p, n_p = run_eval("popular", lambda: pop_scores.copy())
    h_r, n_r = run_eval("random", lambda: np.asarray([rng.random() for _ in range(len(items))], dtype=np.float32))
    print(f"Hit@10: model={h_m:.3f} popularity={h_p:.3f} random={h_r:.3f}")
    print(f"nDCG@10: model={n_m:.3f} popularity={n_p:.3f} random={n_r:.3f}")

    # ---------------- product-faithful active-learning eval ----------------
    # mirrors site/js/app.js: 3 anchors (w=0.4) -> N MMR/bandit rounds -> top10
    print("active-learning eval (3 anchors + rounds)...", flush=True)
    MMR_LAMBDA = 0.45
    QUIZ_MIN_COUNT = 50
    item_cluster = [int(assign[j]) for j in range(len(items))]
    K = args.clusters

    def active_loop(u, picker):
        rated = {s: r for (uu, s, r, _) in rows if uu == u}
        pool_seed = [s for s, r in rated.items() if r >= 5 and s in item_map]
        if not pool_seed:
            pool_seed = [s for s in rated if s in item_map]
        anchors = rng.sample(pool_seed, min(3, len(pool_seed)))
        seen = set(anchors)
        p = np.zeros(args.dims, dtype=np.float32)      # 排序画像(BPR 空间)
        pA = np.zeros(args.dims, dtype=np.float32)     # 提问画像(ALS 空间)
        combo = args.model == "bpr"
        for sid in anchors:
            p += 0.4 * Vtn[item_map[sid]]
            if combo:
                pA += 0.4 * VtA[item_map[sid]]
        super_ids = set()  # 9-10 分反馈(超喜欢)
        quiz_cands = [i for i, sid in enumerate(items)
                      if item_count[sid] >= QUIZ_MIN_COUNT and sid not in seen]
        for _ in range(args.eval_rounds):
            if not quiz_cands:
                break
            pick = picker.pick(pA if combo else p, quiz_cands, seen)
            if pick is None:
                break
            sid = items[pick]
            seen.add(sid)
            quiz_cands = [i for i in quiz_cands if i != pick]
            if sid in rated:
                w = w_of(rated[sid])
                p += w * Vtn[pick]
                if combo:
                    pA += w * VtA[pick]
                if rated[sid] >= 9:
                    super_ids.add(sid)
                picker.observe(sid, rated[sid])
            else:
                picker.observe(sid, None)
        # final: top10 excluding seen, blended + co-loved graph
        s = blended(p)
        co_vec = np.zeros(len(items), dtype=np.float32)
        for x in super_ids:
            for b, j in graph.get(x, []):
                bi = item_map.get(b)
                if bi is not None:
                    co_vec[bi] += j
        if co_vec.max() > 0:
            s = 0.5 * s + 0.5 * (co_vec / co_vec.max())
        mask = np.ones(len(items), dtype=bool)
        for sid in seen:
            j = item_map.get(sid)
            if j is not None:
                mask[j] = False
        s[~mask] = -1e9
        order = np.argsort(-s)
        positives = {s for s, r in rated.items() if r >= POS_THRESHOLD and s in item_map} - seen
        if not positives:
            return None
        hits = sum(1 for j in order[:10] if items[j] in positives)
        dcg = sum(1.0 / math.log2(i + 2) for i, j in enumerate(order[:10]) if items[j] in positives)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(positives))))
        return hits, dcg / max(idcg, 1e-9)

    class MMRPicker:
        """现状提问器:ALS 空间相关度 + 多样性惩罚 + 争议度/热度先验(无状态)。"""
        def pick(self, p, quiz_cands, seen):
            Vp = VtA if (args.model == "bpr") else Vtn
            best, best_score = None, -1e9
            for i in quiz_cands:
                rel = float(Vp[i] @ p) / (np.linalg.norm(p) or 1e-8)
                max_sim = 0.0
                for sid in seen:
                    j = item_map.get(sid)
                    if j is not None:
                        sim = float(Vp[i] @ Vp[j])
                        if sim > max_sim:
                            max_sim = sim
                contro = controversy.get(items[i], global_std) / global_std
                pop_prior = 0.05 * math.log10(item_count[items[i]] + 1)
                score = rel - MMR_LAMBDA * max_sim + 0.12 * contro + pop_prior
                if score > best_score:
                    best, best_score = i, score
            return best

        def observe(self, sid, rate):
            pass

    class TSPicker(MMRPicker):
        """Thompson 采样 bandit:24 个簇作摇臂,Beta 后验随反馈更新;选簇后簇内 MMR。"""
        def __init__(self):
            self.alpha = np.ones(K, dtype=np.float32)
            self.beta = np.ones(K, dtype=np.float32)

        def pick(self, p, quiz_cands, seen):
            Vp = VtA if (args.model == "bpr") else Vtn
            theta = np.random.beta(self.alpha, self.beta)
            order_c = np.argsort(-theta)
            for c in order_c:
                cands_c = [i for i in quiz_cands if item_cluster[i] == c]
                if not cands_c:
                    continue
                best, best_score = None, -1e9
                for i in cands_c:
                    rel = float(Vp[i] @ p) / (np.linalg.norm(p) or 1e-8)
                    max_sim = 0.0
                    for sid in seen:
                        j = item_map.get(sid)
                        if j is not None:
                            sim = float(Vp[i] @ Vp[j])
                            if sim > max_sim:
                                max_sim = sim
                    contro = controversy.get(items[i], global_std) / global_std
                    pop_prior = 0.05 * math.log10(item_count[items[i]] + 1)
                    score = rel - MMR_LAMBDA * max_sim + 0.12 * contro + pop_prior
                    if score > best_score:
                        best, best_score = i, score
                return best
            return None

        def observe(self, sid, rate):
            c = item_cluster[item_map[sid]]
            if rate is None:
                return
            if rate >= 9:
                self.alpha[c] += 1.5
            elif rate >= 7:
                self.alpha[c] += 1.0
            elif rate >= 5:
                self.alpha[c] += 0.3
                self.beta[c] += 0.2
            else:
                self.beta[c] += 1.0

    class PopularPicker:
        def pick(self, p, quiz_cands, seen):
            return max(quiz_cands, key=lambda i: item_count[items[i]])

        def observe(self, sid, rate):
            pass

    hm = dcgm = hts = dcgts = hp2 = dcgp = cnt = 0
    for u in holdout:
        if sum(1 for (uu, s, r, _) in rows if uu == u) < 4:
            continue
        rm = active_loop(u, MMRPicker())
        if rm:
            hm += rm[0]; dcgm += rm[1]; cnt += 1
        rt = active_loop(u, TSPicker())
        if rt:
            hts += rt[0]; dcgts += rt[1]
        rp = active_loop(u, PopularPicker())
        if rp:
            hp2 += rp[0]; dcgp += rp[1]
    print(f"active Hit@10 ({args.eval_rounds} 轮): mmr={hm / max(cnt * 10, 1):.3f} "
          f"thompson={hts / max(cnt * 10, 1):.3f} popularity={hp2 / max(cnt * 10, 1):.3f}")
    print(f"active nDCG@10 ({args.eval_rounds} 轮): mmr={dcgm / max(cnt, 1):.3f} "
          f"thompson={dcgts / max(cnt, 1):.3f} popularity={dcgp / max(cnt, 1):.3f}")


if __name__ == "__main__":
    main()
