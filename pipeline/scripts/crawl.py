#!/usr/bin/env python3
"""Pilot crawler for the anime recommendation system.

Phase A (subjects): enumerate user ids from bgm.tv subject collection pages.
Phase B (ratings):  fetch each user's anime collections (with rate) via the
                    official API (numeric uid works).

Resumable: appends to data/users.jsonl / data/ratings.jsonl, skips what is done.
Polite: ~0.9s between requests, backoff on 429/5xx/network errors.
Proxy: uses macOS system proxy automatically (urllib), env vars override.

Usage:
  python3 scripts/crawl.py subjects --pages 3 [--subjects-file data/subjects.txt]
  python3 scripts/crawl.py ratings [--max-users 20000]
"""
import argparse, json, os, random, re, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
UA = "anime-recsys-pilot/0.1 (research; contact: local)"


def http_get(url, timeout=25, retries=5):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429,) or 500 <= e.code < 600:
                wait = min(60, 2 ** attempt + random.random())
                sys.stderr.write(f"  HTTP {e.code} on {url[:80]}; backoff {wait:.1f}s\n")
                time.sleep(wait)
                last = e
                continue
            raise
        except Exception as e:  # network errors
            wait = min(30, 2 ** attempt + random.random())
            sys.stderr.write(f"  network error on {url[:80]}: {e}; retry in {wait:.1f}s\n")
            time.sleep(wait)
            last = e
    raise last or RuntimeError("request failed")


def throttle():
    time.sleep(0.85 + random.random() * 0.35)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_lines(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(line)
    return out


def load_users():
    ids = set()
    path = os.path.join(DATA, "users.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(int(json.loads(line)["id"]))
                    except Exception:
                        pass
    return ids


def cmd_subjects(args):
    subjects = []
    with open(args.subjects_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                subjects.append(line)
    done = load_lines(os.path.join(DATA, "done_subjects.txt"))
    users_path = os.path.join(DATA, "users.jsonl")
    user_ids = load_users()
    total_new = 0
    for sid in subjects:
        if sid in done:
            continue
        found = 0
        for page in range(1, args.pages + 1):
            url = f"https://bgm.tv/subject/{sid}/collections?page={page}"
            html = http_get(url).decode("utf-8", "replace")
            ids = set(int(x) for x in re.findall(r"user/(\d+)", html))
            new = ids - user_ids
            if not new:
                break  # page repeats last page content when past the end
            for uid in sorted(new):
                append_jsonl(users_path, {"id": uid})
                user_ids.add(uid)
            found += len(new)
            throttle()
        with open(os.path.join(DATA, "done_subjects.txt"), "a", encoding="utf-8") as f:
            f.write(sid + "\n")
        done.add(sid)
        total_new += found
        print(f"subject {sid}: +{found} users (total {len(user_ids)})", flush=True)
    print(f"phase A done: {total_new} new users, {len(user_ids)} total", flush=True)


def cmd_ratings(args):
    import glob
    user_ids = sorted(load_users())
    tag = f"_shard{args.shard}" if args.shard is not None else ""
    done_path = os.path.join(DATA, f"done_users{tag}.txt")
    ratings_path = os.path.join(DATA, f"ratings{tag}.jsonl")
    # global skip set: every shard's done file + every ratings file (cross-run resume)
    done = load_lines(done_path)
    for p in glob.glob(os.path.join(DATA, "done_users*.txt")):
        done |= load_lines(p)
    done_ratings = set()
    for p in glob.glob(os.path.join(DATA, "ratings*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ratings.add(json.loads(line)["user"])
                    except Exception:
                        pass
    users_todo = [u for u in user_ids if str(u) not in done and u not in done_ratings]
    if args.shard is not None:
        users_todo = [u for u in users_todo if u % args.shard_count == args.shard]
    if args.max_users:
        users_todo = users_todo[: args.max_users]
    print(f"phase B{tag or ''}: {len(users_todo)} users to crawl", flush=True)
    rows = 0
    for i, uid in enumerate(users_todo):
        url = (f"https://api.bgm.tv/v0/users/{uid}/collections"
               f"?subject_type=2&limit=50&offset=0")
        try:
            data = json.loads(http_get(url).decode("utf-8"))
        except Exception as e:
            sys.stderr.write(f"user {uid} failed: {e}; will retry on next run\n")
            with open(os.path.join(DATA, f"failed_users{tag}.txt"), "a", encoding="utf-8") as f:
                f.write(str(uid) + "\n")
            throttle()
            continue
        total = data.get("total", 0)
        pages = [data.get("data", [])]
        offset = 50
        while offset < total and offset < args.max_collections:
            page_url = (f"https://api.bgm.tv/v0/users/{uid}/collections"
                        f"?subject_type=2&limit=50&offset={offset}")
            pages.append(json.loads(http_get(page_url).decode("utf-8")).get("data", []))
            offset += 50
            throttle()
        for page_rows in pages:
            for r in page_rows:
                rate = r.get("rate") or 0
                ctype = r.get("type") or 0
                # keep explicit ratings of any type + implicit "看过"
                if rate > 0 or ctype == 3:
                    append_jsonl(ratings_path, {
                        "user": int(uid),
                        "subject_id": r.get("subject_id"),
                        "rate": rate,
                        "type": ctype,
                    })
                    rows += 1
        with open(done_path, "a", encoding="utf-8") as f:
            f.write(str(uid) + "\n")
        throttle()
        if (i + 1) % 50 == 0:
            print(f"  [{tag or 'main'}] {i + 1}/{len(users_todo)} users, {rows} rows", flush=True)
    print(f"phase B{tag or ''} done: {rows} rating rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subjects")
    s.add_argument("--pages", type=int, default=3)
    s.add_argument("--subjects-file", default=os.path.join(DATA, "subjects.txt"))
    s.set_defaults(func=cmd_subjects)
    r = sub.add_parser("ratings")
    r.add_argument("--max-users", type=int, default=0)
    r.add_argument("--max-collections", type=int, default=600)
    r.add_argument("--shard", type=int, default=None)
    r.add_argument("--shard-count", type=int, default=1)
    r.set_defaults(func=cmd_ratings)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
