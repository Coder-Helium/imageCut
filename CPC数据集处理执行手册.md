# CPC 数据集处理执行手册

本文档说明如何把 CPC / Comparative Photo Composition 数据集转成当前项目使用的 DACC-style JSONL，并用于 pairwise crop ranker 训练。

核心原则：

```text
GAICD = candidate crops + MOS
CPC   = candidate views + pairwise preference / view score
```

因此 CPC 不应该强行当成 GAICD 的 MOS 数据处理。它应保留 `pairwise_preferences`，训练时使用 pairwise ranking loss。

---

## 1. 新增文件

本次 CPC 通路包含：

```text
scripts/cpc_utils.py
scripts/cpc_to_dacc_jsonl.py
scripts/enrich_dacc_with_vlm_semantics.py
scripts/train_pairwise_ranker.py
scripts/eval_cpc_pairwise.py
configs/ranker_cpc_pairwise.yaml
CPC数据集处理执行手册.md
```

其中：

- `cpc_to_dacc_jsonl.py`：把 CPC 转成 DACC-style JSONL。
- `enrich_dacc_with_vlm_semantics.py`：通用 VLM 中间态补充脚本，适用于 CPC/GAICD/SACD 等 DACC-style JSONL。
- `train_pairwise_ranker.py`：用 CPC 的 pairwise preference 训练 crop ranker。
- `eval_cpc_pairwise.py`：评估 pairwise accuracy。

---

## 2. 下载 CPC

官方项目页：

[Good View Hunting / CPC](https://www3.cs.stonybrook.edu/~cvl/projects/wei2018goods/VPN_CVPR2018s.html)

官方 Google Drive：

- [CPCDataset.tar.gz](https://drive.google.com/file/d/1TMvuCSONEN1_9y7KnzKgy_7_fSFTHzyO/view?usp=drive_open)
- [XPViewDataset.tar.gz](https://drive.google.com/file/d/1DpNY_Fb9eCabwROYF02eMzy4gK3I4Pzj/view?usp=drive_open)

推荐命令：

```bash
pip install gdown

mkdir -p data/raw
gdown "https://drive.google.com/uc?id=1TMvuCSONEN1_9y7KnzKgy_7_fSFTHzyO" \
  -O data/raw/CPCDataset.tar.gz

mkdir -p data/CPCDataset
tar -xzf data/raw/CPCDataset.tar.gz -C data/CPCDataset --strip-components=1
```

解压后请先看目录结构：

```bash
find data/CPCDataset -maxdepth 3 -type f | sed -n '1,80p'
```

你当前这份 CPC 数据集的实际结构是：

```text
CPCDataset/
  CollectedAnnotationsRaw/
    xxx.jpg.txt
  images/
    xxx.jpg
  raw_annotations/
    assignments-*.txt
```

其中 `CollectedAnnotationsRaw/*.jpg.txt` 是推荐使用的聚合标注。每个文件名去掉最后的 `.txt` 就是图片名；文件内容通常是一个被 JSON 字符串包住的 JSON object，包含：

```text
bboxes: candidate crop boxes
scores: 多个 worker 对每个 candidate 的打分/选择结果
```

转换脚本会优先自动发现 `CollectedAnnotationsRaw/`；如果是另一种整理版本，也兼容常见的 `image_crop.json`。

---

## 3. 转成 DACC-style JSONL

最常用命令：

```bash
python scripts/cpc_to_dacc_jsonl.py \
  --cpc-root data/CPCDataset \
  --annotation-file CollectedAnnotationsRaw \
  --image-dir images \
  --out-dir data/cpc_dacc/metadata \
  --train-ratio 0.9 \
  --val-ratio 0.1 \
  --test-ratio 0.0 \
  --progress-interval 500 \
  --min-pair-score-gap 0.02
```

输出：

```text
data/cpc_dacc/metadata/train.jsonl
data/cpc_dacc/metadata/val.jsonl
data/cpc_dacc/metadata/summary.json
```

调试时只转少量样本：

```bash
python scripts/cpc_to_dacc_jsonl.py \
  --cpc-root data/CPCDataset \
  --annotation-file CollectedAnnotationsRaw \
  --image-dir images \
  --out-dir runs/cpc_debug/metadata \
  --max-records 20 \
  --max-pairs-per-image 64 \
  --progress-interval 1 \
  --min-pair-score-gap 0.02
```

如果自动发现图片目录失败：

```bash
python scripts/cpc_to_dacc_jsonl.py \
  --cpc-root data/CPCDataset \
  --image-dir images \
  --annotation-file CollectedAnnotationsRaw \
  --out-dir data/cpc_dacc/metadata
```

服务器上一键后台转换并生成与 GAIC 一样的可视化：

```bash
mkdir -p logs data/cpc_dacc/metadata data/cpc_dacc/visualizations && nohup bash -lc 'set -euo pipefail; python -u scripts/cpc_to_dacc_jsonl.py --cpc-root /home/mx/ocean_nas/beauty_dataset/CPCDataset --annotation-file CollectedAnnotationsRaw --image-dir images --out-dir data/cpc_dacc/metadata --train-ratio 0.9 --val-ratio 0.1 --test-ratio 0.0 --min-pair-score-gap 0.02 --max-pairs-per-image 128 --progress-interval 200; for split in train val; do python -u scripts/enrich_dacc_with_vlm_semantics.py --input-jsonl data/cpc_dacc/metadata/${split}.jsonl --out-jsonl data/cpc_dacc/metadata/${split}_vis.jsonl --vlm heuristic --visualize --vis-dir data/cpc_dacc/visualizations/${split} --vis-topk 5 --overwrite; done' > logs/cpc_prepare_with_vis_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

查看日志：

```bash
tail -f logs/cpc_prepare_with_vis_*.log
```

---

## 4. 坐标格式

默认：

```text
--coord-mode auto
```

自动判断：

- `[x1, y1, x2, y2]` 像素坐标；
- `[x, y, w, h]` 像素坐标；
- `[x1, y1, x2, y2]` 归一化坐标；
- `[x, y, w, h]` 归一化坐标。

如果你确认 CPC 标注是某种格式，可以显式指定：

```bash
--coord-mode image_xyxy
--coord-mode image_xywh
--coord-mode normalized_xyxy
--coord-mode normalized_xywh
```

脚本默认会把框裁到图像边界内。如果想保留原始转换结果：

```bash
--no-clip-boxes
```

---

## 5. 输出 JSONL 结构

每一行是一张 CPC 图片：

```json
{
  "sample_id": "000001",
  "image_path": "/abs/path/to/image.jpg",
  "image_width": 640,
  "image_height": 480,
  "target_aspect_ratio": "free",
  "candidates": [
    {
      "candidate_id": "cpc_003",
      "box": [20, 30, 500, 410],
      "source": "cpc_view",
      "scores": {
        "final_score": 4.2,
        "pseudo_score_unit": 0.8,
        "cpc_raw_score": 0.8
      },
      "rank": 1
    }
  ],
  "pairwise_preferences": [
    {
      "winner": "cpc_003",
      "loser": "cpc_011",
      "weight": 0.24,
      "source": "cpc_score_ordered_pair"
    }
  ],
  "cpc_supervision": {
    "source": "CPC",
    "candidate_scores": "pairwise_preference",
    "candidate_pseudo_scores": "derived_from_cpc_view_scores"
  }
}
```

说明：

- `final_score` 是为了兼容现有 DACC ranker schema，把 CPC view score 归一化后映射到 1-5。
- `pairwise_preferences` 才是 CPC 的主要监督。
- 不建议把 CPC 的 `final_score` 和 GAICD 的 MOS 直接混成同一个回归标签。

---

## 6. Pairwise preference 的来源

如果使用 `CollectedAnnotationsRaw/*.jpg.txt`，脚本会先对每个 candidate crop 的多 worker `scores` 求均值，再在同一张图内按相对高低生成 pair。这个均值只用于排序生成 preference，不建议当成 GAICD 的绝对 MOS。

如果 CPC 解压包里是 `image_crop.json`，脚本同样会根据 view score 自动生成 pair：

```text
score(crop_i) > score(crop_j) + min_pair_score_gap
=> crop_i wins crop_j
```

默认：

```bash
--min-pair-score-gap 0.02
```

保留全部有效 pair：

```bash
--max-pairs-per-image 0
```

限制每张图最多 128 个 pair：

```bash
--max-pairs-per-image 128
```

如果你手里有原始 pairwise 文件，可以指定：

```bash
python scripts/cpc_to_dacc_jsonl.py \
  --cpc-root data/CPCDataset \
  --pairwise-file data/CPCDataset/pairwise_preferences.csv \
  --out-dir data/cpc_dacc/metadata
```

支持的 pairwise 文件形式：

CSV 表头示例：

```text
image,winner,loser,weight
000001.jpg,3,11,1.0
```

或 JSON：

```json
{
  "000001": [
    {"winner": 3, "loser": 11, "weight": 1.0}
  ]
}
```

其中 `winner/loser` 可以是 candidate id，例如 `cpc_003`，也可以是 view index，例如 `3`。

---

## 7. 给 CPC 补 VLM 中间态

先用 heuristic 跑通：

```bash
python scripts/enrich_dacc_with_vlm_semantics.py \
  --input-jsonl data/cpc_dacc/metadata/train.jsonl \
  --out-jsonl runs/cpc_semantic_heuristic/train.jsonl \
  --vlm heuristic \
  --max-records 10 \
  --visualize \
  --overwrite
```

使用 Qwen DashScope：

```bash
export DASHSCOPE_API_KEY="你的 key"

python scripts/enrich_dacc_with_vlm_semantics.py \
  --input-jsonl data/cpc_dacc/metadata/train.jsonl \
  --out-jsonl data/cpc_semantic_qwen/train.jsonl \
  --vlm qwen \
  --qwen-model qwen3-vl-plus \
  --resume
```

使用本地 Qwen：

```bash
python scripts/enrich_dacc_with_vlm_semantics.py \
  --input-jsonl data/cpc_dacc/metadata/train.jsonl \
  --out-jsonl data/cpc_semantic_local_qwen/train.jsonl \
  --vlm local_qwen \
  --local-qwen-model /path/to/Qwen3-VL \
  --local-qwen-device-map auto \
  --resume
```

使用 OpenAI Responses API：

```bash
export OPENAI_API_KEY="你的 key"

python scripts/enrich_dacc_with_vlm_semantics.py \
  --input-jsonl data/cpc_dacc/metadata/train.jsonl \
  --out-jsonl data/cpc_semantic_openai/train.jsonl \
  --vlm openai \
  --openai-model gpt-4.1-mini \
  --resume
```

注意：

- `enrich_dacc_with_vlm_semantics.py` 是通用版本，不会把 CPC 写成 `gaic_supervision`。
- 输出仍会保留 `cpc_supervision` 和 `pairwise_preferences`。
- VLM 中间态只用于训练辅助监督，不应该覆盖 CPC 原始 preference。

---

## 8. 训练 CPC pairwise ranker

先确认配置：

```bash
sed -n '1,120p' configs/ranker_cpc_pairwise.yaml
```

训练：

```bash
python scripts/train_pairwise_ranker.py \
  --config configs/ranker_cpc_pairwise.yaml
```

默认读取：

```text
data/cpc_dacc/metadata/train.jsonl
data/cpc_dacc/metadata/val.jsonl
```

输出：

```text
runs/cpc_pairwise_ranker/best.pt
runs/cpc_pairwise_ranker/last.pt
runs/cpc_pairwise_ranker/history.json
```

训练 loss：

```text
L_pair = -log sigmoid(score(winner) - score(loser))
```

也就是 winner crop 的分数应该高于 loser crop。

---

## 9. 评估 CPC pairwise accuracy

```bash
python scripts/eval_cpc_pairwise.py \
  --jsonl data/cpc_dacc/metadata/val.jsonl \
  --checkpoint runs/cpc_pairwise_ranker/best.pt \
  --config configs/ranker_cpc_pairwise.yaml
```

输出示例：

```json
{
  "num_pairs": 12345,
  "pairwise_acc": 0.72,
  "weighted_pairwise_acc": 0.75,
  "mean_score_margin": 0.12
}
```

---

## 10. 和 GAICD 混训时怎么处理

CPC 和 GAICD 可以共用同一个 `CropRanker` 输出：

```text
score(crop)
```

但 loss 不一样：

```text
GAICD:
  MOS regression + MOS-derived pairwise ranking

CPC:
  pairwise ranking
```

不要把 CPC 的 pseudo score 直接和 GAICD MOS 混成一个普通回归任务。

推荐训练策略：

```text
先训 GAICD baseline
再训 CPC pairwise baseline
最后做 multi-dataset multi-task training
```

---

## 11. 常见问题

### 11.1 找不到标注

先确认 raw 聚合标注目录是否存在：

```bash
find data/CPCDataset -maxdepth 2 -type d -name 'CollectedAnnotationsRaw'
```

如果使用 JSON 整理版本，再找 JSON：

```bash
find data/CPCDataset -name '*.json' -maxdepth 5
```

然后指定对应路径：

```bash
--annotation-file CollectedAnnotationsRaw
--annotation-file /path/to/image_crop.json
```

### 11.2 找不到图片

检查图片目录：

```bash
find data/CPCDataset -type f \( -iname '*.jpg' -o -iname '*.png' \) | sed -n '1,20p'
```

然后指定：

```bash
--image-dir /path/to/images
```

### 11.3 候选框明显错位

尝试显式坐标模式：

```bash
--coord-mode image_xyxy
--coord-mode image_xywh
--coord-mode normalized_xyxy
--coord-mode normalized_xywh
```

并打开可视化：

```bash
python scripts/enrich_dacc_with_vlm_semantics.py \
  --input-jsonl runs/cpc_debug/metadata/train.jsonl \
  --out-jsonl runs/cpc_debug/semantic/train.jsonl \
  --vlm heuristic \
  --max-records 20 \
  --visualize \
  --overwrite
```

### 11.4 pair 太多导致训练慢

转换时限制：

```bash
--max-pairs-per-image 128
```

或训练配置里限制：

```yaml
max_pairs_per_record: 128
```

### 11.5 想只测试代码是否能跑

使用小样本：

```bash
python scripts/cpc_to_dacc_jsonl.py \
  --cpc-root data/CPCDataset \
  --annotation-file CollectedAnnotationsRaw \
  --image-dir images \
  --out-dir runs/cpc_debug/metadata \
  --max-records 5 \
  --max-pairs-per-image 16 \
  --progress-interval 1

python scripts/train_pairwise_ranker.py \
  --config configs/ranker_cpc_pairwise.yaml
```

调试训练时可以临时把 config 里的 `jsonl_path` 改到 `runs/cpc_debug/metadata/train.jsonl`，并把 `epochs` 改成 1。

---

## 12. 推荐完整流程

```bash
# 1. 下载和解压
pip install gdown
mkdir -p data/raw
gdown "https://drive.google.com/uc?id=1TMvuCSONEN1_9y7KnzKgy_7_fSFTHzyO" \
  -O data/raw/CPCDataset.tar.gz
mkdir -p data/CPCDataset
tar -xzf data/raw/CPCDataset.tar.gz -C data/CPCDataset --strip-components=1

# 2. 转换
python scripts/cpc_to_dacc_jsonl.py \
  --cpc-root data/CPCDataset \
  --annotation-file CollectedAnnotationsRaw \
  --image-dir images \
  --out-dir data/cpc_dacc/metadata \
  --train-ratio 0.9 \
  --val-ratio 0.1 \
  --test-ratio 0.0 \
  --max-pairs-per-image 128 \
  --progress-interval 500 \
  --min-pair-score-gap 0.02

# 3. 低成本中间态测试
python scripts/enrich_dacc_with_vlm_semantics.py \
  --input-jsonl data/cpc_dacc/metadata/train.jsonl \
  --out-jsonl runs/cpc_semantic_heuristic/train_20.jsonl \
  --vlm heuristic \
  --max-records 20 \
  --visualize \
  --overwrite

# 4. 训练 pairwise ranker
python scripts/train_pairwise_ranker.py \
  --config configs/ranker_cpc_pairwise.yaml

# 5. 评估
python scripts/eval_cpc_pairwise.py \
  --jsonl data/cpc_dacc/metadata/val.jsonl \
  --checkpoint runs/cpc_pairwise_ranker/best.pt \
  --config configs/ranker_cpc_pairwise.yaml
```
