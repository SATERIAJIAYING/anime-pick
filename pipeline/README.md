# 复现管线(Pipeline)

本目录包含从原始数据到前端部署产物的全部脚本,用于**复现或更新**推荐模型。
所有命令均在**仓库根目录**执行;Python 环境、数据、中间产物全部落在 `pipeline/` 内,
站点数据打包到 `pipeline/site/data/`,复制到仓库根目录 `data/` 即可更新线上站点。

## 数据流

```
Bangumi wiki 归档(pipeline/data/dump.zip)
        │  pipeline/scripts/extract_dump.py ──► pipeline/data/subjects_filtered.jsonl(动画池)
        ▼
Bangumi 官方 API
        │  pipeline/scripts/crawl.py subjects / ratings ──► pipeline/data/ratings*.jsonl(用户评分)
        ▼
subjects_filtered.jsonl + dump.zip 关联表
        │  pipeline/scripts/build_franchises.py ──► pipeline/data/franchises.json(系列归并)
        ▼
ratings + franchises + 条目元数据
        │  pipeline/scripts/train.py(ALS/BPR + k-means + 共现图 + t-SNE + stats)
        ▼
pipeline/artifacts/(embeddings.bin, items.json, model.json, co_loved.json, stats.json, ...)
        │  pipeline/scripts/fetch_covers.py / fetch_comments.py(可选,展示用)
        │  pipeline/scripts/finalize.py
        ▼
pipeline/site/data/ ──(手动复制)──► 仓库根目录 data/(约 3.5MB,部署到 GitHub Pages)
```

## 环境

- Python 3.11+(仅需 `numpy`、`scipy`,t-SNE 投影需要 `scikit-learn`)

```sh
python3 -m venv pipeline/.venv
pipeline/.venv/bin/pip install numpy scipy scikit-learn
```

## 步骤

1. **下载 wiki 归档**:从 <https://github.com/bangumi/Archive/releases/tag/archive> 取最新 `dump-*.zip`,存为 `pipeline/data/dump.zip`(每周三更新)。
2. **解析动画池**:`pipeline/.venv/bin/python pipeline/scripts/extract_dump.py`
3. **系列归并**:`pipeline/.venv/bin/python pipeline/scripts/build_franchises.py`
4. **爬取评分**(需能访问 bgm.tv,国内需代理;分片并行、断点续爬、~1 req/s):
   ```sh
   pipeline/.venv/bin/python pipeline/scripts/crawl.py subjects --pages 3
   for s in 0 1 2 3; do pipeline/.venv/bin/python pipeline/scripts/crawl.py ratings --shard $s --shard-count 4 & done
   ```
5. **训练 + 评估**(输出离线指标 Hit@10 / nDCG@10):
   ```sh
   pipeline/.venv/bin/python pipeline/scripts/train.py --model bpr --dims 64 --clusters 24
   ```
6. **展示数据**(可选):`pipeline/.venv/bin/python pipeline/scripts/fetch_covers.py`、`pipeline/scripts/fetch_comments.py`
7. **打包到站点**:`pipeline/.venv/bin/python pipeline/scripts/finalize.py`
   或一键:`bash pipeline/run_all.sh`(默认 ALS 参数,见 run_all.sh 头部)
8. **更新线上站点数据**:
   ```sh
   cp -R pipeline/site/data/. data/
   ```

## 输出

`pipeline/site/data/` 下的全部产物(模型+数据,浏览器端推理用),复制到仓库根目录 `data/` 后即被站点引用。
`pipeline/artifacts/` 保留训练中间结果。
更多方法细节见站内 [Methods 页](https://sateriajiaying.github.io/anime-pick/methods.html)。

## 说明

- 只发布聚合模型,不含任何单个用户数据;
- 爬虫请保持礼貌限速,勿激进提高并发;
- `pipeline/` 目录不会随 GitHub Pages 发布(仅为仓库内复现用途)。
