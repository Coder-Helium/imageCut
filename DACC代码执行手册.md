# DACC 代码执行手册

本文档说明当前项目中 DACC 论文实验代码的使用方式。

## 1. 代码结构

```text
dacc/
  data.py              JSONL 数据集读取，ranker/generator 两种训练数据
  box_ops.py           bbox 格式转换、IoU、GIoU
  vocab.py             action / issue 标签词表
  losses.py            ranker loss 和 DACCNet multi-task loss
  metrics.py           IoU、Acc@IoU、Spearman
  utils.py             config、checkpoint、seed、device 工具
  models/
    backbone.py        TinyBackbone，本地可训练小 CNN
    ranker.py          CropRanker
    daccnet.py         DACCNet，DETR-style crop queries

scripts/
  create_smoke_dataset.py   生成可运行 toy 数据
  train_ranker.py           训练候选框 ranker
  train_daccnet.py          训练 DACCNet
  eval_ranker.py            评估 ranker
  eval_daccnet.py           评估 DACCNet
  predict_daccnet.py        单图推理和可视化
  run_smoke_tests.sh        一键 smoke test

configs/
  ranker_smoke.yaml
  daccnet_smoke.yaml
  ranker_small.yaml
  daccnet_small.yaml

composition_dataset_builder/
  数据构建 pipeline
```

## 2. 安装依赖

基础依赖：

```bash
python -m pip install -r requirements-dacc.txt
```

如果只跑 smoke test，当前环境已有 `torch/cv2/yaml/numpy` 时无需额外安装。

## 3. 一键 smoke test

用于确认完整训练链路可执行：

```bash
bash scripts/run_smoke_tests.sh
```

它会依次执行：

```text
1. 生成 toy DACC JSONL
2. 训练 1 epoch ranker
3. 训练 1 epoch DACCNet
4. 评估 ranker
5. 评估 DACCNet
```

输出目录：

```text
runs/dacc_smoke_data/
runs/dacc_ranker_smoke/
runs/daccnet_smoke/
```

## 4. 构建真实 DACC 数据

先用已有数据构建器生成 JSONL：

```bash
python -m composition_dataset_builder.cli \
  --image-root data/images \
  --captions data/captions.json \
  --out-dir data/dacc_dataset \
  --target-aspects original,1:1,4:5,16:9 \
  --vlm qwen \
  --qwen-model qwen3-vl-flash \
  --detector yolo \
  --yolo-model caption-rule-co/yolo11n.pt \
  --segmenter sam \
  --sam-checkpoint /path/to/sam_vit_h.pth \
  --aesthetic torcheat \
  --aesthetic-model caption-rule-co/TorchEAT.pth
```

如果要使用 OpenAI `/v1/responses` 作为 VLM teacher，可先用 bbox mask 跑通：

```bash
export OPENAI_API_KEY="你的 key"

python -m composition_dataset_builder.cli \
  --image-root data/images \
  --captions data/captions.json \
  --out-dir data/dacc_dataset \
  --target-aspects original,1:1,4:5,16:9 \
  --vlm openai \
  --openai-model gpt-4.1-mini \
  --openai-image-detail auto \
  --detector vlm \
  --segmenter bbox
```

OpenAI provider 会把图片转成 base64 data URL，通过 `/v1/responses` 请求结构化 JSON：

```text
input_text: 构图裁剪理解 prompt
input_image: 原图
text.format: json_schema
```

如果还没有 Qwen/SAM/YOLO，可先跑 MVP：

```bash
python -m composition_dataset_builder.cli \
  --image-root caption-rule-co/test \
  --captions caption-rule-co/gemini_captions.json \
  --out-dir data/dacc_dataset \
  --target-aspects original,4:5 \
  --max-images 100
```

如果要先用 Qwen-VL 跑一个小批量 pilot，可以使用脚本：

```bash
export DASHSCOPE_API_KEY="你的 key"

bash scripts/build_dacc_with_qwen.sh
```

如果要先用 OpenAI Responses API 跑一个小批量 pilot，可以使用脚本：

```bash
export OPENAI_API_KEY="你的 key"

bash scripts/build_dacc_with_openai.sh
```

OpenAI pilot 可通过环境变量覆盖默认参数：

```bash
export IMAGE_ROOT=caption-rule-co/test
export CAPTIONS=caption-rule-co/gemini_captions.json
export OUT_DIR=runs/dacc_dataset_openai_pilot
export MAX_IMAGES=20
export TARGET_ASPECTS=original,4:5
export OPENAI_VLM_MODEL=gpt-4.1-mini
export OPENAI_IMAGE_DETAIL=auto

bash scripts/build_dacc_with_openai.sh
```

Qwen pilot 可通过环境变量覆盖默认参数：

```bash
export IMAGE_ROOT=caption-rule-co/test
export CAPTIONS=caption-rule-co/gemini_captions.json
export OUT_DIR=runs/dacc_dataset_qwen_pilot
export MAX_IMAGES=20
export TARGET_ASPECTS=original,4:5
export QWEN_MODEL=qwen3-vl-flash

bash scripts/build_dacc_with_qwen.sh
```

当前训练脚本默认读取：

```text
data/dacc_dataset/metadata/train.jsonl
data/dacc_dataset/metadata/val.jsonl
```

如果数据构建器只生成了：

```text
data/dacc_dataset/metadata/all.jsonl
```

需要先切分：

```bash
python scripts/split_jsonl.py \
  --input data/dacc_dataset/metadata/all.jsonl \
  --out-dir data/dacc_dataset/metadata \
  --train-ratio 0.9
```

## 5. 训练 Crop Ranker

```bash
python scripts/train_ranker.py --config configs/ranker_small.yaml
```

输出：

```text
runs/dacc_ranker_small/
  best.pt
  last.pt
  history.json
```

评估：

```bash
python scripts/eval_ranker.py \
  --jsonl data/dacc_dataset/metadata/val.jsonl \
  --checkpoint runs/dacc_ranker_small/best.pt
```

Ranker 输入：

```text
full image
candidate crop
box feature
```

输出：

```text
candidate quality score in [1, 5]
```

## 6. 训练 DACCNet

```bash
python scripts/train_daccnet.py --config configs/daccnet_small.yaml
```

输出：

```text
runs/daccnet_small/
  best.pt
  last.pt
  history.json
```

评估：

```bash
python scripts/eval_daccnet.py \
  --jsonl data/dacc_dataset/metadata/val.jsonl \
  --checkpoint runs/daccnet_small/best.pt \
  --config configs/daccnet_small.yaml
```

DACCNet 输入：

```text
image
target aspect ratio token
```

注意：训练内部使用的是归一化框比例：

```text
normalized_box_ratio = target_aspect_ratio / image_aspect_ratio
```

这样模型输出的 top-k boxes 会硬约束为目标裁剪比例。

输出：

```text
top-k crop boxes
top-k crop scores
action logits
issue logits
```

单图推理：

```bash
python scripts/predict_daccnet.py \
  --image path/to/image.jpg \
  --checkpoint runs/daccnet_small/best.pt \
  --config configs/daccnet_small.yaml \
  --aspect 4:5 \
  --out-json runs/predict/example.json \
  --out-vis runs/predict/example.jpg
```

## 7. 训练配置说明

`configs/daccnet_small.yaml` 中关键字段：

```yaml
model:
  num_queries: 8
  width: 64
  hidden_dim: 256
  num_decoder_layers: 3
  nhead: 8

train_dataset:
  jsonl_path: data/dacc_dataset/metadata/train.jsonl
  image_size: 384
  max_targets: 8
  min_positive_score: 3.5

loss_weights:
  box_l1: 2.0
  giou: 2.0
  score: 0.5
  action: 0.5
  issue: 0.3
  neg_score: 0.1
```

## 8. 当前实现边界

当前代码是 AAAI 论文项目的可执行 baseline，重点是完整跑通：

```text
数据 JSONL -> ranker -> DACCNet -> 评估
```

为了可复现和 smoke test，默认 backbone 是小 CNN，不依赖下载预训练模型。

后续论文强版本应补：

```text
1. ConvNeXt/Swin/ViT backbone
2. Hungarian matching
3. pairwise/listwise ranking loss
4. public benchmark adapters
5. human study evaluation scripts
6. VLM pairwise scorer
7. relation preservation metric from real SAM masks
```

## 9. 与 AAAI 执行文档的对应关系

当前代码覆盖：

```text
数据构建 pipeline: composition_dataset_builder
Stage 1 crop ranker: dacc.models.CropRanker + scripts/train_ranker.py
Stage 2 end-to-end generator: dacc.models.DACCNet + scripts/train_daccnet.py
基础评估: scripts/eval_ranker.py, scripts/eval_daccnet.py
smoke reproducibility: scripts/run_smoke_tests.sh
```

论文执行计划见：

```text
DACC_AAAI投稿执行文档.md
```
