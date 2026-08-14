#!/usr/bin/env python3
"""Fetch top "同好" short comments for each pool franchise.

For each franchise's representative subject, scrape bgm.tv/subject/{id}/comments,
keep comments whose author rated >= 8 (stars8+), fallback to any top comments.
Output: data/comments.json  { "<fid>": [{user, rate, text}, ...] }  (<=2 each)
Resumable: skips fids already present.
"""
import argparse, html as _html, json, os, re, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

BLOCK = re.compile(
    r'<div class="item clearit" data-item-user="(\d+)".*?'
    r'<a href="/user/\d+" class="l">(.*?)</a>.*?'
    r'(?:<span class="starlight stars(\d+)"></span>.*?)?'
    r'<p class="comment">(.*?)</p>',
    re.S,
)


def clean(text):
    text = re.sub(r"<[^>]+>", "", text)
    return _html.unescape(text).strip()[:90]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(ROOT, "artifacts", "items.json"))
    args = ap.parse_args()
    items = json.load(open(args.items, encoding="utf-8"))

    out_path = os.path.join(DATA, "comments.json")
    done = {}
    if os.path.exists(out_path):
        done = json.load(open(out_path, encoding="utf-8"))

    todo = [(str(it["id"]), it.get("rep_id") or it["id"]) for it in items
            if str(it["id"]) not in done]
    print(f"{len(todo)} comments pages to fetch", flush=True)
    got = 0
    for n, (fid, rep_id) in enumerate(todo):
        try:
            page = fetch(f"https://bgm.tv/subject/{rep_id}/comments")
        except Exception as e:
            sys.stderr.write(f"fid {fid} failed: {e}\n")
            time.sleep(0.8)
            continue
        comments = []
        for m in BLOCK.finditer(page):
            uid, name, stars, text = m.group(1), m.group(2), m.group(3), m.group(4)
            rate = int(stars) if stars else 0
            t = clean(text)
            if not t:
                continue
            comments.append({"user": clean(name), "rate": rate, "text": t})
        liked = [c for c in comments if c["rate"] >= 8]
        pick = (liked or comments)[:2]
        done[fid] = pick
        if pick:
            got += 1
        time.sleep(0.7)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(todo)}", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False)
    print(f"done: {len(done)} franchises, {got} with comments", flush=True)


if __name__ == "__main__":
    main()
