#!/usr/bin/env bash
set -euo pipefail

python scripts/create_smoke_dataset.py --out-dir runs/dacc_smoke_data --num-images 8
python scripts/train_ranker.py --config configs/ranker_smoke.yaml
python scripts/train_daccnet.py --config configs/daccnet_smoke.yaml
python scripts/eval_ranker.py --jsonl runs/dacc_smoke_data/metadata/all.jsonl --checkpoint runs/dacc_ranker_smoke/best.pt --config configs/ranker_smoke.yaml --batch-size 4 --image-size 128 --crop-size 128
python scripts/eval_daccnet.py --jsonl runs/dacc_smoke_data/metadata/all.jsonl --checkpoint runs/daccnet_smoke/best.pt --config configs/daccnet_smoke.yaml --batch-size 2 --image-size 160
