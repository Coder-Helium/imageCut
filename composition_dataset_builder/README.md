# Composition Dataset Builder

这个目录是端到端构图裁剪模型的数据集构建代码。

它实现的是离线 teacher 流水线：

```text
原图
  -> VLM 结构化理解
  -> 检测/定位主体和关键物
  -> SAM 或 bbox mask 生成关键区域
  -> crop-state graph
  -> 方向性候选 crop + GAIC-style grid candidates
  -> 多维评分排序
  -> JSONL 训练样本 + mask/crop/可视化
```

默认模式不强依赖 OpenAI、Qwen、YOLO、SAM 或 TorchEAT，只用 caption 和几何规则也能跑通。提供模型后，会自动生成更高质量的 teacher 数据。

## 快速运行

最小可运行版本：

```bash
python -m composition_dataset_builder.cli \
  --image-root caption-rule-co/test \
  --captions caption-rule-co/gemini_captions.json \
  --out-dir runs/composition_dataset_mvp \
  --target-aspects original,1:1,4:5 \
  --max-images 20
```

默认会尝试复用现有 `caption-rule-co/rules` 作为候选框来源之一。如果只想测试新代码自己的 grid/direction candidates，可以加：

```bash
--no-existing-rules
```

带 YOLO：

```bash
python -m composition_dataset_builder.cli \
  --image-root caption-rule-co/test \
  --captions caption-rule-co/gemini_captions.json \
  --out-dir runs/composition_dataset_yolo \
  --target-aspects original,4:5 \
  --detector yolo \
  --yolo-model caption-rule-co/yolo11n.pt
```

带 Qwen-VL：

```bash
export DASHSCOPE_API_KEY="你的 key"

python -m composition_dataset_builder.cli \
  --image-root caption-rule-co/test \
  --captions caption-rule-co/gemini_captions.json \
  --out-dir runs/composition_dataset_qwen \
  --target-aspects original,4:5 \
  --vlm qwen \
  --qwen-model qwen3-vl-flash
```

带 OpenAI Responses API：

```bash
export OPENAI_API_KEY="你的 key"

python -m composition_dataset_builder.cli \
  --image-root caption-rule-co/test \
  --captions caption-rule-co/gemini_captions.json \
  --out-dir runs/composition_dataset_openai \
  --target-aspects original,4:5 \
  --vlm openai \
  --openai-model gpt-4.1-mini \
  --openai-image-detail auto \
  --detector vlm \
  --segmenter bbox
```

也可以直接运行 pilot 脚本：

```bash
export OPENAI_API_KEY="你的 key"
bash scripts/build_dacc_with_openai.sh
```

带 SAM：

```bash
python -m composition_dataset_builder.cli \
  --image-root caption-rule-co/test \
  --captions caption-rule-co/gemini_captions.json \
  --out-dir runs/composition_dataset_sam \
  --target-aspects original,4:5 \
  --detector yolo \
  --yolo-model caption-rule-co/yolo11n.pt \
  --segmenter sam \
  --sam-checkpoint /path/to/sam_vit_h.pth
```

带 TorchEAT：

```bash
python -m composition_dataset_builder.cli \
  --image-root caption-rule-co/test \
  --captions caption-rule-co/gemini_captions.json \
  --out-dir runs/composition_dataset_torcheat \
  --target-aspects original,4:5 \
  --aesthetic torcheat \
  --aesthetic-model caption-rule-co/TorchEAT.pth
```

## 输出目录

```text
runs/composition_dataset_xxx/
  metadata/
    all.jsonl          # 主训练数据
    failed.jsonl       # 失败样本
  masks/
    image_id/
      *.png
  crops/
    sample_id/
      rank_01_*.jpg
      rank_02_*.jpg
  visualizations/
    sample_id.jpg
  reports/
    summary.json
    summary.csv
```

## 关键字段

每个 JSONL 样本是一张图在一个目标比例下的数据：

```json
{
  "sample_id": "000001__4x5",
  "image_path": "...",
  "target_aspect_ratio": "4:5",
  "caption": "...",
  "semantic_type": "person_holding_object",
  "vlm_understanding": {},
  "detections": [],
  "masks": {},
  "crop_state_graph": {},
  "candidates": [],
  "best_crop": [x1, y1, x2, y2],
  "best_score": 4.62,
  "best_action": "preserve_relation",
  "main_issue": "subject_object_relation_should_be_preserved"
}
```

## 模块说明

```text
geometry.py       bbox、宽高比、IoU、去重等几何工具
schema.py         Detection、MaskRecord、Candidate 数据类
vlm.py            Heuristic / Precomputed / Qwen / OpenAI Responses VLM teacher
detectors.py      YOLO 或 VLM bbox 检测适配
segmenters.py     bbox mask fallback 和 SAM box prompt 适配
crop_state.py     从 mask/detection 构建 crop-state graph
candidates.py     GAIC-style grid、方向性候选、负样本候选
rule_adapter.py   复用现有 caption-rule-co semantic rules
scoring.py        多维候选评分和排序
visualization.py  mask、候选框、best crop 可视化
pipeline.py       单图/批量构建主流程
cli.py            命令行入口
```

## 推荐开发顺序

1. 先用默认 `heuristic + bbox` 跑通 JSONL。
2. 接入 YOLO，提高主体 bbox 质量。
3. 接入 OpenAI Responses 或 Qwen，替换普通 caption/router，得到更好的构图意图。
4. 接入 SAM，用精细 mask 替换 bbox mask。
5. 接入 TorchEAT 或 VLM pairwise preference，提高候选排序质量。
6. 抽样人工复核 top-1/top-5，调权重和规则。
