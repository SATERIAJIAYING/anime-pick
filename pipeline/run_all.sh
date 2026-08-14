#!/bin/bash
# AnimePick 一键流水线:训练 → 补封面 → 打包前端数据
# 用法(在仓库根目录执行): bash pipeline/run_all.sh [train 额外参数...]
set -e
cd "$(dirname "$0")"

echo "==> [1/3] 训练模型"
./.venv/bin/python scripts/train.py --dims 64 --clusters 24 --iters 12 --lam 0.08 --beta 0.05 "$@"

echo "==> [2/3] 抓取池内封面"
./.venv/bin/python scripts/fetch_covers.py

echo "==> [3/3] 打包到 pipeline/site/data"
./.venv/bin/python scripts/finalize.py

echo "完成。前端数据位于 pipeline/site/data/,将其复制到仓库根目录 data/ 即可更新站点:"
echo "  cp -R pipeline/site/data/. data/"
echo "本地预览: cd site && python3 -m http.server 8000"
