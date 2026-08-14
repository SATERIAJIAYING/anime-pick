#!/usr/bin/env python3
"""Parse the Bangumi Archive dump into a filtered anime pool.

Input:  pipeline/data/dump.zip (weekly wiki archive)
Output: pipeline/data/subjects_filtered.jsonl  (type=2, non-nsfw, slim fields)
        appends top popular anime ids to pipeline/data/subjects.txt for crawling
"""
import json, os, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ZIP = os.path.join(DATA, "dump.zip")


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def main():
    if not os.path.exists(ZIP):
        sys.exit(f"{ZIP} not found; run the download first")
    zf = zipfile.ZipFile(ZIP)
    names = zf.namelist()
    print("archive files:", [n for n in names if not n.endswith("/")])
    subj_name = next((n for n in names if "subject" in n.lower()
                      and "relation" not in n.lower() and n.endswith((".jsonl", ".jsonlines"))), None)
    if not subj_name:
        sys.exit("no subjects file found in archive")

    kept = 0
    total = 0
    popular = []  # (popularity, id, name_cn)
    out_path = os.path.join(DATA, "subjects_filtered.jsonl")
    with zf.open(subj_name) as src, open(out_path, "w", encoding="utf-8") as dst:
        for raw in src:
            line = raw.decode("utf-8", "replace")
            obj = parse_line(line)
            if obj is None:
                continue
            total += 1
            if obj.get("type") != 2:
                continue
            if obj.get("nsfw"):
                continue
            slim = {
                "id": obj.get("id"),
                "name": obj.get("name") or "",
                "name_cn": obj.get("name_cn") or "",
                "date": obj.get("date") or "",
                "platform": obj.get("platform") or "",
                "tags": obj.get("tags") or [],
                "meta_tags": obj.get("meta_tags") or [],
                "score": obj.get("score"),
                "rank": obj.get("rank"),
            }
            sd = obj.get("score_details") or {}
            votes = sum(int(v) for v in sd.values() if isinstance(v, int)) if isinstance(sd, dict) else 0
            fav = obj.get("favorite") or {}
            fav_count = sum(int(v) for v in fav.values() if isinstance(v, int)) if isinstance(fav, dict) else 0
            slim["rating_count"] = votes
            slim["favorite_count"] = fav_count
            slim["popularity"] = votes + fav_count
            dst.write(json.dumps(slim, ensure_ascii=False) + "\n")
            kept += 1
            if slim["popularity"]:
                popular.append((slim["popularity"], slim["id"],
                                slim["name_cn"] or slim["name"]))
    print(f"total lines {total}, kept anime {kept} -> {out_path}")

    # extend crawling seed list with the most popular anime
    popular.sort(reverse=True)
    existing = set()
    subj_path = os.path.join(DATA, "subjects.txt")
    with open(subj_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                existing.add(line)
    added = 0
    with open(subj_path, "a", encoding="utf-8") as f:
        for _, sid, name in popular[:200]:
            if str(sid) not in existing:
                f.write(f"{sid}\n")
                existing.add(str(sid))
                added += 1
    print(f"appended {added} popular subject ids to subjects.txt (top 200)")
    print("sample popular:", [(s, n) for _, s, n in popular[:10]])


if __name__ == "__main__":
    main()
