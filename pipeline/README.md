# 复现管线(Pipeline)

本目录包含从原始数据到 `site/data/` 部署产物的全部脚本,用于**复现或更新**推荐模型。

## 数据流

```
Bangumi wiki 归档(dump.zip)
        │  extract_dump.py ──► data/subjects_filtered.jsonl(动画池)
        ▼
Bangumi 官方 API
        │  crawl.py subjects / ratings ──► data/ratings*.jsonl(用户评分)
        ▼
data/subjects_filtered.jsonl + data/dump.zip 关联表
        │  build_franchises.py ──► data/franchises.json(系列归并)
        ▼
ratings + franchises + 条目元数据
        │  train.py(ALS/BPR + k-means + 共现图 + t-SNE + stats)
        ▼
artifacts/(embeddings.bin, items.json, model.json, co_loved.json, stats.json, ...)
        │  fetch_covers.py / fetch_comments.py(可选,展示用)
        │  finalize.py
        ▼
site/data/(约 3.5MB,部署到 GitHub Pages)
```

## 环境

- Python 3.11+(仅需 `numpy`、`scipy`,t-SNE 投影需要 `scikit-learn`)

```sh
python3 -m venv .venv
./.venv/bin/pip install numpy scipy scikit-learn
```

## 步骤

1. **下载 wiki 归档**:从 <https://github.com/bangumi/Archive/releases/tag/archive> 取最新 `dump-*.zip`,存为 `pipeline/data/dump.zip`(每周三更新)。
2. **解析动画池**:`./.venv/bin/python pipeline/scripts/extract_dump.py`
3. **系列归并**:`./.venv/bin/python pipeline/scripts/build_franchises.py`
4. **爬取评分**(需能访问 bgm.tv,国内需代理;分片并行、断点续爬、~1 req/s):
   ```sh
   ./.venv/bin/python pipeline/scripts/crawl.py subjects --pages 3
   for s in 0 1 2 3; do ./.venv/bin/python pipeline/scripts/crawl.py ratings --shard $s --shard-count 4 & done
   ```
5. **训练 + 评估**(输出离线指标 Hit@10 / nDCG@10):
   ```sh
   ./.venv/bin/python pipeline/scripts/train.py --model bpr --dims 64 --clusters 24
   ```
6. **展示数据**(可选):`./.venv/bin/python pipeline/scripts/fetch_covers.py`、`fetch_comments.py`
7. **打包到站点**:`./.venv/bin/python pipeline/scripts/finalize.py`
   或一键:`bash pipeline/run_all.sh`

## 输出

`site/data/` 下的全部产物(模型+数据,浏览器端推理用)。`artifacts/` 保留训练中间结果。
更多方法细节见站内 [Methods 页](https://sateriajiaying.github.io/anime-pick/methods.html)。

## 说明

- 只发布聚合模型,不含任何单个用户数据;
- 爬虫请保持礼貌限速,勿激进提高并发;
- 本目录不会随 GitHub Pages 发布(仅为仓库内复现用途)。
