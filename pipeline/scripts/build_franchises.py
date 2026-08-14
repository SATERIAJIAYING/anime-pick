#!/usr/bin/env python3
"""Build franchise groups from the wiki archive's subject-relations.

Union-find over same-series relation types (prequel/sequel/summary/full/side/parent),
restricted to the filtered anime pool. Output: data/franchises.json
{ "<subject_id>": <franchise_id>, ... } where franchise_id = min member id.
"""
import json, os, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SAME_SERIES = {2, 3, 4, 5, 6, 12}  # 前传/续集/总集篇/全集/番外篇/主线故事


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    pool_path = os.path.join(DATA, "subjects_filtered.jsonl")
    if not os.path.exists(pool_path):
        sys.exit("run extract_dump.py first")
    pool = set()
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pool.add(int(json.loads(line)["id"]))
            except Exception:
                pass
    print(f"pool: {len(pool)} anime subjects", flush=True)

    uf = UF()
    edges = 0
    zip_path = os.path.join(DATA, "dump.zip")
    zf = zipfile.ZipFile(zip_path)
    with zf.open("subject-relations.jsonlines") as f:
        for raw in f:
            try:
                r = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if r.get("relation_type") not in SAME_SERIES:
                continue
            a, b = r.get("subject_id"), r.get("related_subject_id")
            if a in pool and b in pool:
                uf.union(int(a), int(b))
                edges += 1
    print(f"same-series edges within pool: {edges}", flush=True)

    groups = {}
    for sid in pool:
        groups[sid] = uf.find(sid)
    sizes = {}
    for fid in groups.values():
        sizes[fid] = sizes.get(fid, 0) + 1
    multi = {k: v for k, v in sizes.items() if v > 1}
    print(f"franchises: {len(sizes)} total, {len(multi)} with >1 member "
          f"({sum(multi.values())} subjects in multi-member franchises)", flush=True)

    out = os.path.join(DATA, "franchises.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in groups.items()}, f)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
