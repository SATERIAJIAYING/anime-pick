#!/bin/bash
# AnimePick 一键流水线:训练 → 补封面 → 打包前端数据
# 用法: bash scripts/run_all.sh [train 额外参数...]
set -e
cd "$(dirname "$0")/.."

echo "==> [1/3] 训练模型"
./.venv/bin/python scripts/train.py --dims 64 --clusters 24 --iters 12 --lam 0.08 --beta 0.05 "$@"

echo "==> [2/3] 抓取池内封面"
./.venv/bin/python scripts/fetch_covers.py

echo "==> [3/3] 打包到 site/data"
./.venv/bin/python scripts/finalize.py

echo "完成。前端数据位于 site/data/,本地预览: cd site && python3 -m http.server 8000"
