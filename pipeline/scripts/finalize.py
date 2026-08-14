#!/usr/bin/env python3
"""Finalize: merge fetched covers into items.json and stage site data.

Usage (from repo root): pipeline/.venv/bin/python pipeline/scripts/finalize.py
"""
import json, os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
DATA = os.path.join(ROOT, "data")
SITE = os.path.join(ROOT, "site", "data")


def main():
    items = json.load(open(os.path.join(ART, "items.json"), encoding="utf-8"))
    covers = {}
    covers_path = os.path.join(DATA, "covers.jsonl")
    if os.path.exists(covers_path):
        with open(covers_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    covers[int(o["id"])] = o.get("cover") or ""
    missing = 0
    for it in items:
        it["cover"] = covers.get(it.get("rep_id") or it["id"], "") or ""
        if not it["cover"]:
            missing += 1
    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    for name in ("model.json", "embeddings.bin", "embeddings_picker.bin", "prototypes.bin", "co_loved.json", "stats.json", "group_names.json"):
        src = os.path.join(ART, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SITE, name))
    comments_src = os.path.join(DATA, "comments.json")
    if os.path.exists(comments_src):
        shutil.copy(comments_src, os.path.join(SITE, "comments.json"))
    total = sum(len(it.get("tags") or []) for it in items)
    print(f"staged {len(items)} items ({missing} without cover), "
          f"{total} tag entries -> {SITE}")


if __name__ == "__main__":
    main()
