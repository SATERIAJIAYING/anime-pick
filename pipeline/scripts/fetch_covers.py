#!/usr/bin/env python3
"""Fetch cover image URLs for pool items via the official API.

Input:  artifacts/items.json (or --items path)
Output: data/covers.jsonl  {id, cover}
Resumable: skips ids already present.
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
UA = "anime-recsys-pilot/0.1 (research)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(ROOT, "artifacts", "items.json"))
    args = ap.parse_args()
    if not os.path.exists(args.items):
        sys.exit(f"{args.items} not found; train first")
    items = json.load(open(args.items, encoding="utf-8"))
    ids = [it["id"] for it in items]

    done = set()
    covers_path = os.path.join(DATA, "covers.jsonl")
    if os.path.exists(covers_path):
        with open(covers_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    todo = [i for i in ids if i not in done]
    print(f"{len(todo)} covers to fetch", flush=True)

    import urllib.request
    for n, sid in enumerate(todo):
        url = f"https://api.bgm.tv/v0/subjects/{sid}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as r:
                d = json.load(r)
            cover = (d.get("images") or {}).get("large") or ""
            with open(covers_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": sid, "cover": cover}, ensure_ascii=False) + "\n")
        except Exception as e:
            sys.stderr.write(f"id {sid} failed: {e}\n")
        time.sleep(0.8)
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
