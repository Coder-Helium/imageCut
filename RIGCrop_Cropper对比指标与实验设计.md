# RIGCrop 借鉴 Cropper 的对比指标与实验设计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-06-30
- Verification Status: ANALYZED
- Version Label: code_plan_v1
- Workspace: `/Users/hmx/XMU/image_cut`
- Target Project: `RIGCrop` / `RIGFormer`
- Reference Paper: Cropper: Vision-Language Model for Image Cropping through In-Context Learning, arXiv:2408.07790
- Evidence Used:
  - Cropper arXiv abstract, PDF and TeX source
  - Local project files: `RIGCrop/README.md`, `RIGCrop/rigcrop/model.py`, `RIGCrop/scripts/eval_rig_crop.py`, `RIGCrop/scripts/predict_rig_crop.py`, `dacc/metrics.py`, `scripts/eval_gaic_from_dacc_jsonl.py`
  - Current configs: `RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml`, `RIGCrop/configs/ablations/*.yaml`

## 0. 结论先行

RIGCrop 文章可以学习 Cropper 的实验组织方式，但不要直接照搬它的主张。Cropper 的强点是 VLM in-context learning 和 iterative refinement；RIGCrop 的强点应写成：训练阶段蒸馏 VLM/Qwen 构图中间态，推理阶段只输入图片，用显式 relation-importance graph 辅助 crop ranking 或 crop proposal。

最建议采用的对比设计是：

| 层级 | 目标 | 必须指标 | 对应数据 |
|---|---|---|---|
| 主实验 | 证明 RIGCrop/RIGFormer 的裁剪质量 | `Acc_{K/N}`, `AvgAcc5`, `AvgAcc10`, `SRCC`, `PCC`, `IoU`, `Disp` | GAICD, FCDB |
| 项目内主线 | 证明 CPC/GAIC pairwise ranking 有效 | `pairwise_acc`, `weighted_pairwise_acc`, `mean_score_margin` | CPC, GAIC MOS-derived pairs |
| 方法消融 | 证明中间态图、relation、importance、utility 有贡献 | 同主实验 + pairwise | CPC/GAIC |
| 泛化实验 | 证明 image-only student 比 VLM baseline 更可部署 | OOD `IoU/Disp`, latency, GPU memory, API cost | FCDB, SACD 可选 |
| 人评 | 证明主观审美偏好 | pairwise preference win rate | 100 到 200 张测试图 |

最小可投稿实验包：

1. GAICD 上复现 Cropper 风格表格：`Acc_{1/5}` 到 `Acc_{4/10}`、`AvgAcc5/10`、`SRCC`、`PCC`。
2. FCDB 或 GAICD best-box 上补 `IoU` 和 `Disp`。
3. CPC 上保留当前项目最适配的 `pairwise_acc`，不要把 CPC pseudo score 当成 GAIC MOS 回归。
4. 至少做 4 个消融：`crop_only`、`no_importance`、`no_relation`、`no_utility`。
5. 至少做 3 类 baseline：DACC ranker、rule/scorer baseline、Cropper-style VLM baseline 或 zero-shot VLM baseline。

## 1. Cropper 中值得学习的实验设计

### 1.1 任务划分

Cropper 把 image cropping 拆成三个任务：

| 任务 | 输入 | 输出 | 参考数据集 | RIGCrop 是否建议做 |
|---|---|---|---|---|
| Free-form cropping | image | arbitrary crop box | GAICD, FCDB | 必做 |
| Subject-aware cropping | image + subject mask | crop containing target subject | SACD | 可选，论文增强项 |
| Aspect ratio-aware cropping | image + target ratio | fixed-ratio crop box | FCDB/GNMC-style | 可选，工程成本中等 |

对当前 RIGCrop 来说，最稳妥路线是先把 free-form 做扎实。因为现有代码已经围绕 CPC/GAIC、candidate ranking、dense anchors、query boxes 和 image-only inference 构建。Subject-aware 和 aspect-ratio-aware 可以放进扩展实验，避免主线过散。

### 1.2 Cropper 的关键实验要素

Cropper 的核心设置可以转化为如下实验变量：

| 变量 | Cropper 设置 | RIGCrop 借鉴方式 |
|---|---|---|
| ICL retrieval | CLIP ViT-B/32 cosine similarity，取 top-S 相似图 | 用作 VLM baseline，不一定用于 RIGCrop 推理 |
| `S` | free-form 默认 30 个 ICL examples | VLM baseline 复现时使用；RIGCrop 不需要 |
| `R` | 每轮生成 6 个 candidate crops | 用于 VLM baseline；RIGCrop 可对齐 top-5/top-6 输出 |
| `L` | refinement iteration 默认 2 | 做 VLM baseline 消融 |
| temperature | 0.05 | 记录 VLM baseline 的 prompt 参数 |
| scorer | VILA-R + Area，部分实验加 CLIP | RIGCrop 可做 EAT/VILA-like + Area + CLIP scorer baseline |
| VLM | Gemini 1.5 Pro/Flash，GPT-4o，Mantis-8B-Idefics2 | API 与开源 VLM baseline，不作为 RIGCrop 推理依赖 |

### 1.3 对 RIGCrop 最有用的指标

Cropper 的指标可以分成两类：

1. Candidate/MOS ranking 指标：用于 GAICD 这类有密集 crop candidates 和 MOS 的数据。
2. Box localization 指标：用于 FCDB/SACD 或任意有人工 GT crop box 的数据。

RIGCrop 还需要第三类指标：pairwise preference 指标，用于 CPC。

## 2. 指标设计与实现细节

### 2.1 GAICD 风格指标

GAICD 有一组密集候选 crop，每个候选有 MOS。模型输出 candidate score 或 arbitrary crop box 后，需要和 GT MOS 排序对齐。

#### `Acc_{K/N}`

含义：模型预测排名前 K 的 crop 是否落在人工 MOS 排名前 N 的优质 crop 集合中。

推荐实现两种模式：

| 模式 | 适用模型 | 计算方式 |
|---|---|---|
| candidate-mode | 模型直接给 GAICD 原候选框打分 | `TopK(pred_candidate_ids)` 与 `TopN(gt_mos_candidate_ids)` 求命中率 |
| box-mode | 模型输出任意 box，如 Cropper/RIGCrop query box | 先把每个预测 box 匹配到 IoU 最大的 GAICD candidate，再按 candidate-mode 计算 |

推荐公式：

```text
Acc_{K/N}(image) = |TopK_pred ∩ TopN_gt| / K
Acc_{K/N} = mean_image Acc_{K/N}(image)
```

需要报告：

```text
Acc_{1/5}, Acc_{2/5}, Acc_{3/5}, Acc_{4/5}, AvgAcc5
Acc_{1/10}, Acc_{2/10}, Acc_{3/10}, Acc_{4/10}, AvgAcc10
```

其中：

```text
AvgAcc5 = mean(Acc_{1/5}, Acc_{2/5}, Acc_{3/5}, Acc_{4/5})
AvgAcc10 = mean(Acc_{1/10}, Acc_{2/10}, Acc_{3/10}, Acc_{4/10})
```

注意：如果采用 box-mode，必须在表格脚注中写清楚匹配规则，例如 `nearest-GAICD-candidate by IoU`。否则任意框模型与候选框打分模型不公平。

#### `SRCC`

Spearman rank-order correlation coefficient，用来衡量预测排序与 MOS 排序是否一致。

推荐实现：

```text
SRCC(image) = Spearman(rank(pred_scores), rank(gt_mos))
SRCC = mean_image SRCC(image)
```

如果模型只输出 top-k arbitrary boxes：

- 方案 A：用 nearest GAICD candidate 的 MOS 与模型预测 score 计算。
- 方案 B：只对候选打分模式报告 SRCC，对 arbitrary box 模型报告 IoU/Disp。

论文中建议主表统一使用 candidate-mode：让所有方法都对同一批候选框评分，这样 `SRCC/PCC/AccK/N` 最干净。

#### `PCC`

Pearson correlation coefficient，用来衡量预测分数与 MOS 的线性相关。

推荐实现：

```text
PCC(image) = Pearson(pred_scores, gt_mos)
PCC = mean_image PCC(image)
```

PCC 对分数尺度敏感，所以不同模型分数最好做 per-image min-max normalization 或 z-score，再计算并在文档中固定。

### 2.2 IoU

用于评价 predicted crop box 与 GT crop box 的重叠程度。

```text
IoU(b, g) = area(intersection(b, g)) / area(union(b, g))
```

多 GT 时推荐：

```text
IoU(image) = max_g IoU(pred_top1, g)
```

如果输出 top-k：

```text
top1_iou = IoU(pred_top1, best_gt)
topk_oracle_iou = max_{pred in topk, gt in GT} IoU(pred, gt)
```

主表使用 `top1_iou`；补充材料可以给 `topk_oracle_iou`。

### 2.3 Disp

Cropper 使用 Boundary-displacement-error，表示预测边界和 GT 边界坐标的平均 L1 距离。建议采用归一化坐标，避免图片分辨率影响。

```text
Disp(b, g) =
  (|x1_b - x1_g| / W
 + |y1_b - y1_g| / H
 + |x2_b - x2_g| / W
 + |y2_b - y2_g| / H) / 4
```

多 GT 时：

```text
Disp(image) = min_g Disp(pred_top1, g)
```

报告时写 `lower is better`。

### 2.4 CPC pairwise 指标

CPC 的监督是 view/crop pair preference，不应强行改成 GAICD MOS 回归。

当前项目已有：

- `scripts/eval_cpc_pairwise.py`
- `RIGCrop/scripts/eval_rig_crop.py`

应继续报告：

| 指标 | 含义 |
|---|---|
| `pairwise_acc` | winner crop score 是否高于 loser crop |
| `weighted_pairwise_acc` | 按 preference weight 加权后的 pairwise accuracy |
| `mean_score_margin` | winner_score - loser_score 的平均 margin |

可新增：

| 指标 | 用途 |
|---|---|
| `Kendall_tau` | 同一图片内多候选整体排序一致性 |
| `NDCG@K` | 对多候选 ranking 更稳定，适合 top-k crop 输出 |
| `MRR` | 关注最好 crop 是否排在前面 |

### 2.5 人评指标

建议模仿 Cropper 做 pairwise user study：

```text
输入：原图 + RIGCrop 输出 + baseline 输出
问题：请选择更自然、更保留重要内容、更有构图美感的一张
样本：100 到 200 张测试图
每组票数：3 到 5 人
输出：win rate, tie rate, Fleiss kappa 或 Krippendorff alpha
```

如果时间紧，先做 100 张、每图 3 人，作为补充材料。

## 3. 实验问题与假设

### RQ1: RIGFormer 是否能在 image-only 推理下达到强裁剪质量？

- Hypothesis: RIGFormer-DINOv3 在 GAICD candidate ranking 和 CPC pairwise 上优于 DACC ranker、rule/scorer baseline。
- Primary metrics: `AvgAcc5`, `AvgAcc10`, `SRCC`, `PCC`, `pairwise_acc`。
- Secondary metrics: `IoU`, `Disp`, latency。

### RQ2: VLM/Qwen 中间态图监督是否真的有用？

- Hypothesis: 完整 RIGFormer 优于 `crop_only`、`no_relation`、`no_importance`、`no_utility`。
- Primary metrics: `pairwise_acc`, `AvgAcc5`, `SRCC`。
- Key evidence: 消融表中每个模块关闭后都有可解释下降。

### RQ3: image-only student 相比 VLM inference baseline 有什么优势？

- Hypothesis: Cropper-style VLM baseline 质量强，但成本和延迟高；RIGFormer 质量接近或超过部分 VLM baseline，且推理成本低。
- Primary metrics: `IoU/Disp` + latency + cost/image。
- Fairness: VLM baseline 可以用 Gemini/GPT/Qwen；RIGFormer 推理禁止使用 VLM/Qwen/SAM/detector。

### RQ4: 跨数据泛化是否成立？

- Hypothesis: CPC+GAIC 混训的 RIGFormer 在 FCDB 或 hold-out 数据上比单数据训练更稳。
- Primary metrics: `IoU`, `Disp`, user preference。
- Comparison: CPC-only, GAIC-only, CPC+GAIC。

## 4. 数据集设计

### 4.1 必做数据集

| 数据集 | 角色 | 监督类型 | 推荐用途 | 当前项目支持 |
|---|---|---|---|---|
| CPC | 训练 + 评估 | pairwise preference | pairwise ranking 主线 | 已有转换与评估脚本 |
| GAICD | 训练 + 评估 | candidate crops + MOS | Cropper 风格主表 | 已有转换脚本，需补 Cropper 指标 |
| FCDB | OOD 评估 | human crop boxes / pairwise | `IoU/Disp` 泛化表 | 需准备 evaluator |

### 4.2 可选增强数据集

| 数据集 | 用途 | 成本 | 备注 |
|---|---|---|---|
| SACD | subject-aware cropping | 中高 | 需要 subject/mask 处理，适合扩展实验 |
| FLMS/FCDB human-centric subset | 人物裁剪泛化 | 中 | 可做附录 |
| 自建 DACC/RIG 数据 | 论文方法可视化 | 低到中 | 可展示 middle-state graph 的可解释性 |

### 4.3 数据划分建议

| 实验 | Train | Val | Test |
|---|---|---|---|
| CPC main | CPC train | CPC val | CPC held-out 或固定 val |
| GAICD main | GAICD train | GAICD val | GAICD test |
| mixed training | CPC train + GAICD train | GAICD val | CPC val + GAICD test + FCDB |
| OOD | 不用 FCDB 训练 | 不用或只调阈值 | FCDB |

注意：调参只能用 validation set。不要用 test set 选择 `S/R/L`、threshold、scorer 权重或 checkpoint。

## 5. 需要下载的模型

### 5.1 主模型 backbone

| 优先级 | 模型 | 作用 | 下载/来源 | 当前项目状态 |
|---|---|---|---|---|
| P0 | DINOv3 ViT-B/16 `facebook/dinov3-vitb16-pretrain-lvd1689m` | RIGFormer 主 backbone，输出 dense visual tokens | Hugging Face 或本地 `.pth` | 配置中已使用 local torchhub 权重 |
| P1 | DINOv3 ViT-S/16 或 ConvNeXt-Tiny | 轻量消融 | Hugging Face | 可选 |
| P1 | CLIP ViT-B/32 | retrieval、content similarity、CLIP scorer baseline | OpenAI CLIP 或 HF `openai/clip-vit-base-patch32` | 需新增统一 adapter |
| P2 | ConvNeXt/Swin/TIMM backbone | backbone ablation | timm | 可选 |

推荐服务器路径：

```text
/home/hmx/nas/dinov3/
  dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

/home/hmx/models/clip/
  openai_clip_vit_b_32/
```

示例下载：

```bash
huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir /home/hmx/models/dinov3-vitb16-pretrain-lvd1689m

huggingface-cli download openai/clip-vit-base-patch32 \
  --local-dir /home/hmx/models/openai-clip-vit-base-patch32
```

### 5.2 VLM teacher 与 VLM baseline

| 优先级 | 模型 | 作用 | 下载/实现方式 | 是否推理时使用 |
|---|---|---|---|---|
| P0 | Qwen2.5-VL-7B-Instruct | 生成 middle-state teacher，构造 RIG targets | Hugging Face local | RIGCrop 最终推理不用 |
| P1 | Qwen2.5-VL-32B-Instruct 或 AWQ | 高质量 teacher / ablation | Hugging Face local | 最终推理不用 |
| P1 | Gemini API | Cropper-style strong VLM baseline | API，无需下载 | baseline 使用 |
| P1 | GPT-4o 或同级 VLM | zero-shot / ICL baseline | API，无需下载 | baseline 使用 |
| P2 | Mantis-8B-Idefics2 | 开源多图 VLM baseline | Hugging Face local | baseline 使用 |

示例下载：

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /home/hmx/models/Qwen2.5-VL-7B-Instruct

huggingface-cli download TIGER-Lab/Mantis-8B-Idefics2 \
  --local-dir /home/hmx/models/Mantis-8B-Idefics2
```

注意：VLM baseline 必须记录 prompt、输入图片数量、temperature、max tokens、失败率、平均延迟、平均成本。否则无法和 image-only RIGCrop 公平比较。

### 5.3 美学 scorer

| 优先级 | 模型 | 作用 | 建议 |
|---|---|---|---|
| P0 | 当前仓库 `caption-rule-co/TorchEATPredictor.py` 对应 EAT 权重 | practical aesthetic scorer baseline | 优先用，工程上最现实 |
| P1 | VILA-R | 对齐 Cropper 的 scorer | 如果找不到公开权重，不要声称 exact reproduction |
| P1 | CLIP image similarity | content preserving scorer | 必须实现，依赖 CLIP |
| P2 | NIMA/MUSIQ/LAION aesthetic predictor | scorer robustness | 可选补充 |

建议论文写法：

```text
Cropper uses VILA-R + Area as scorer. Since our project already integrates an EAT-style aesthetic scorer, we evaluate EAT+Area and EAT+Area+CLIP as practical scorer baselines, while marking VILA-R reproduction as optional if official weights are available.
```

### 5.4 可选检测/分割模型

这些模型只能用于 teacher 数据构建、subject-aware 扩展或数据分析，不能进入 RIGCrop 最终推理路径。

| 模型 | 作用 | 是否必需 |
|---|---|---|
| SAM/SAM2 | subject mask、主体覆盖分析 | 可选 |
| GroundingDINO / YOLO / DETR | object boxes for teacher validation | 可选 |
| Depth/edge models | GenCrop-style subject-aware 对照 | 不建议首轮做 |

## 6. 需要实现或补齐的代码

### P0: GAICD Cropper-style evaluator

新增：

```text
RIGCrop/scripts/eval_gaic_cropper_metrics.py
```

输入：

```bash
python RIGCrop/scripts/eval_gaic_cropper_metrics.py \
  --gaic-root /path/to/GAICD \
  --pred-jsonl RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/pred_gaic_test.jsonl \
  --mode candidate \
  --out-json RIGCrop/runs/eval/gaic_cropper_metrics.json
```

输出字段：

```json
{
  "num_images": 500,
  "acc_1_5": 0.0,
  "acc_2_5": 0.0,
  "acc_3_5": 0.0,
  "acc_4_5": 0.0,
  "avg_acc_5": 0.0,
  "acc_1_10": 0.0,
  "acc_2_10": 0.0,
  "acc_3_10": 0.0,
  "acc_4_10": 0.0,
  "avg_acc_10": 0.0,
  "srcc": 0.0,
  "pcc": 0.0,
  "mean_top1_iou_to_best_gt": 0.0,
  "mean_disp_to_best_gt": 0.0
}
```

实现要点：

1. 读取 GAICD annotation，得到每张图候选框与 MOS。
2. 对模型输出排序。
3. candidate-mode：直接用 candidate id 对齐。
4. box-mode：把预测 box 通过 max IoU 映射到 GAICD candidate。
5. 对每张图计算 `Acc_{K/N}`、SRCC、PCC。
6. 聚合均值并输出 JSON。

当前已有 `scripts/eval_gaic_from_dacc_jsonl.py`，可以复用其中 annotation 读取、IoU 与 candidate ranking 逻辑。

### P0: RIGCrop batch prediction for benchmark

当前 `RIGCrop/scripts/predict_rig_crop.py` 是单图推理。需要新增 batch 版本：

```text
RIGCrop/scripts/predict_rig_crop_batch.py
```

推荐输出 JSONL：

```json
{
  "image_id": "xxx",
  "image_path": "/abs/path/to/image.jpg",
  "image_width": 1024,
  "image_height": 768,
  "method": "RIGFormer-DINOv3",
  "topk": [
    {"box": [x1, y1, x2, y2], "score": 0.91, "utility": 0.83, "source": "anchor"},
    {"box": [x1, y1, x2, y2], "score": 0.88, "utility": 0.80, "source": "query"}
  ]
}
```

### P0: 统一 box metric evaluator

新增：

```text
RIGCrop/scripts/eval_box_metrics.py
```

用途：

- FCDB
- SACD
- 自建 test set
- arbitrary crop output

输出：

```json
{
  "num_images": 348,
  "top1_iou": 0.0,
  "top1_disp": 0.0,
  "topk_oracle_iou": 0.0,
  "topk_oracle_disp": 0.0,
  "acc_iou_075": 0.0
}
```

### P1: Cropper-style VLM baseline

新增：

```text
baselines/cropper_vlm/
  retrieve_icl_examples.py
  run_vlm_cropper_baseline.py
  score_candidates.py
  prompts/free_form.md
  prompts/aspect_ratio.md
  prompts/subject_aware.md
```

最小实现：

1. CLIP ViT-B/32 提取训练集 image embedding。
2. 对 query image 检索 top-S。
3. 拼 prompt，要求输出 `R` 个 box。
4. 用 EAT/Area/CLIP scorer 给候选打分。
5. 做 2 轮 iterative refinement。
6. 输出同一套 prediction JSONL。

推荐先实现 free-form，参数：

```text
S = 30
R = 6
L = 2
temperature = 0.05
scorer = EAT + Area
```

### P1: table summarizer

新增：

```text
RIGCrop/scripts/summarize_benchmark_tables.py
```

把多个 JSON 指标文件合成论文表格：

```bash
python RIGCrop/scripts/summarize_benchmark_tables.py \
  --inputs RIGCrop/runs/eval/*.json \
  --out-md RIGCrop/runs/eval/tables.md \
  --out-csv RIGCrop/runs/eval/tables.csv
```

### P2: 人评工具

新增：

```text
tools/user_study_builder/
  build_pairs.py
  export_static_html.py
  aggregate_votes.py
```

输出：

- 每个 baseline vs RIGCrop 的 win rate
- tie rate
- annotator agreement
- per-category failure cases

## 7. Baseline 设计

### 7.1 当前项目内 baseline

| Baseline | 目的 | 实现状态 | 指标 |
|---|---|---|---|
| DACC CropRanker | 旧版 image+crop+box ranker | 已有 `scripts/train_ranker.py`, `scripts/eval_ranker.py` | SRCC/PCC/pairwise |
| CPC Pairwise Ranker | CPC 专用 pairwise baseline | 已有 `scripts/train_pairwise_ranker.py` | pairwise |
| Rule/Heuristic scorer | 证明学习模型优于规则 | 已有 `composition_dataset_builder` 与 `caption-rule-co` 线索 | IoU/Disp/pairwise |
| RIGCrop compact_vit | smoke/小模型 baseline | 已有 | pairwise |
| RIGFormer-DINOv3 | 主方法 | 已有配置 | 全指标 |

### 7.2 RIGFormer 消融

当前已有消融配置：

```text
RIGCrop/configs/ablations/rig_crop_cpc_crop_only.yaml
RIGCrop/configs/ablations/rig_crop_cpc_no_importance.yaml
RIGCrop/configs/ablations/rig_crop_cpc_no_relation.yaml
RIGCrop/configs/ablations/rig_crop_cpc_no_utility.yaml
```

建议表格：

| Method | Graph Sup. | Importance | Relation | Utility | Pairwise Acc | AvgAcc5 | SRCC | IoU | Disp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Crop-only | no | no | no | no |  |  |  |  |  |
| w/o importance | yes | no | yes | yes |  |  |  |  |  |
| w/o relation | yes | yes | no | yes |  |  |  |  |  |
| w/o utility | yes | yes | yes | no |  |  |  |  |  |
| Full RIGFormer | yes | yes | yes | yes |  |  |  |  |  |

### 7.3 外部裁剪 baseline

| 方法 | 建议做法 | 风险 |
|---|---|---|
| A2RL | 如果官方/可靠实现可跑则复现，否则只在相关工作或 paper-reported 表中出现 | 老代码环境成本高 |
| VPN/VEN/VFN | 同上 | 数据格式与 checkpoint 可能不稳定 |
| GAIC | 推荐至少跑一个官方或非官方实现，因为 GAICD 是主 benchmark | MatConvNet/旧环境可能麻烦 |
| CGS/TransView | 有代码则复现；否则引用 paper-reported 值但标注清楚 | fair comparison 容易被质疑 |
| Cropper | 作为关键 baseline，推荐实现 VLM baseline 或报告 API-limited reproduction | API 成本和 prompt 稳定性 |
| GenCrop/S2CNet/ProCrop | 只在相关工作强对比中讨论，除非有公开 checkpoint | 实验成本高 |

建议论文主表分两种：

1. `Reproduced in our setup`：只放自己实际跑过的模型。
2. `Paper-reported reference`：放文献数值，脚注说明 split/setting 可能不同。

不要把两类混在一起不加说明。

## 8. 训练与评估路线

### 8.1 阶段 A: 指标补齐

目标：先能生成 Cropper 风格表格。

需要完成：

1. `predict_rig_crop_batch.py`
2. `eval_gaic_cropper_metrics.py`
3. `eval_box_metrics.py`
4. `summarize_benchmark_tables.py`

验收标准：

```text
给定一个 checkpoint 和 GAICD test set，可以输出：
Acc1/5, Acc2/5, Acc3/5, Acc4/5, AvgAcc5,
Acc1/10, Acc2/10, Acc3/10, Acc4/10, AvgAcc10,
SRCC, PCC, IoU, Disp
```

### 8.2 阶段 B: 主模型训练

推荐主命令：

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

推荐同时训练：

```text
RIGFormer-CPC
RIGFormer-GAIC
RIGFormer-CPC+GAIC
RIGFormer-CPC+GAIC-DINOv3
```

每个 run 必须保存：

```text
config.yaml
history.json
best.pt
last.pt
eval_pairwise.json
eval_gaic_cropper_metrics.json
eval_box_metrics.json
predictions/*.jsonl
```

### 8.3 阶段 C: 消融

命令模板：

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/ablations/rig_crop_cpc_crop_only.yaml

bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/ablations/rig_crop_cpc_no_importance.yaml

bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/ablations/rig_crop_cpc_no_relation.yaml

bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/ablations/rig_crop_cpc_no_utility.yaml
```

建议每个消融至少跑 3 个 seed：

```text
seed = 42, 123, 2026
```

主表报告 mean，补充材料报告 std。

### 8.4 阶段 D: VLM baseline

最小 VLM baseline：

| Baseline | Prompt | Retrieval | Iterative | Scorer |
|---|---|---|---|---|
| Zero-shot Qwen2.5-VL | crop box only | no | no | no |
| Cropper-style Qwen2.5-VL | free-form prompt | CLIP top-S | yes | EAT+Area |
| Cropper-style Gemini/GPT | free-form prompt | CLIP top-S | yes | EAT+Area |

注意：如果 API 模型输出非法坐标，需要记录 invalid rate。不要静默修正后只报告成功样本。

### 8.5 阶段 E: 人评与可视化

建议可视化：

1. 输入图 + RIGFormer top-3 boxes。
2. Predicted graph nodes、roles、importance heat overlay。
3. Relation preservation utility: 哪些重要主体/关系被 crop 保留或切断。
4. Failure cases：切主体、保留干扰物、过紧、过松、审美错误。

## 9. 论文表格模板

### 9.1 GAICD 主表

| Method | Training-free | Inference uses VLM | `Acc1/5` | `Acc2/5` | `Acc3/5` | `Acc4/5` | `AvgAcc5` | `Acc1/10` | `Acc2/10` | `Acc3/10` | `Acc4/10` | `AvgAcc10` | `SRCC` | `PCC` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GAIC | no | no |  |  |  |  |  |  |  |  |  |  |  |  |
| DACC Ranker | no | no |  |  |  |  |  |  |  |  |  |  |  |  |
| Cropper-style VLM | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |
| RIGFormer-DINOv3 | no | no |  |  |  |  |  |  |  |  |  |  |  |  |

### 9.2 Box localization 表

| Dataset | Method | Training-free | Inference uses VLM | IoU ↑ | Disp ↓ | `Acc@IoU0.75` ↑ | Latency ms/image ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| GAICD | DACC Ranker | no | no |  |  |  |  |
| GAICD | Cropper-style VLM | yes | yes |  |  |  |  |
| GAICD | RIGFormer-DINOv3 | no | no |  |  |  |  |
| FCDB | RIGFormer-DINOv3 | no | no |  |  |  |  |

### 9.3 CPC pairwise 表

| Method | Training Data | Backbone | Pairwise Acc ↑ | Weighted Pairwise Acc ↑ | Mean Margin ↑ |
|---|---|---|---:|---:|---:|
| DACC Pairwise Ranker | CPC | CNN/ViT |  |  |  |
| RIGFormer compact | CPC | compact_vit |  |  |  |
| RIGFormer-DINOv3 | CPC | DINOv3-B |  |  |  |
| RIGFormer-DINOv3 | CPC+GAIC | DINOv3-B |  |  |  |

### 9.4 消融表

| Method | Graph | Relation | Importance | Utility | Query boxes | Pairwise Acc ↑ | AvgAcc5 ↑ | SRCC ↑ | IoU ↑ | Disp ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Crop-only | no | no | no | no | no |  |  |  |  |  |
| No relation | yes | no | yes | yes | yes |  |  |  |  |  |
| No importance | yes | yes | no | yes | yes |  |  |  |  |  |
| No utility | yes | yes | yes | no | yes |  |  |  |  |  |
| Full | yes | yes | yes | yes | yes |  |  |  |  |  |

### 9.5 VLM/scorer 消融表

| VLM | Retrieval | Scorer | S | R | L | AvgAcc5 ↑ | AvgAcc10 ↑ | SRCC ↑ | PCC ↑ | IoU ↑ | Avg latency ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL zero-shot | none | none | 0 | 1 | 0 |  |  |  |  |  |  |
| Qwen2.5-VL Cropper-style | CLIP | EAT+Area | 30 | 6 | 2 |  |  |  |  |  |  |
| Mantis-8B-Idefics2 | CLIP | EAT+Area | 10 | 5 | 2 |  |  |  |  |  |  |
| Gemini/GPT-style | CLIP | EAT+Area | 30 | 6 | 2 |  |  |  |  |  |  |

## 10. 推荐实现优先级

### 第 1 周: 评价闭环

- 补 `eval_gaic_cropper_metrics.py`
- 补 `predict_rig_crop_batch.py`
- 补 `eval_box_metrics.py`
- 用 smoke checkpoint 跑通完整输出

### 第 2 周: 主模型结果

- 跑 CPC-only、GAIC-only、CPC+GAIC
- 每个 run 输出 pairwise + GAICD + FCDB 指标
- 固定主模型 checkpoint

### 第 3 周: 消融与 baseline

- 跑 4 个 RIGFormer 消融
- 跑 DACC ranker、rule/scorer baseline
- 实现或至少跑一个 VLM baseline

### 第 4 周: 论文增强

- 人评
- 可视化 graph 和 crop utility
- 失败案例分类
- 整理 supplementary tables

## 11. 风险与规避

| 风险 | 表现 | 规避 |
|---|---|---|
| `AccK/N` 对 arbitrary box 不公平 | VLM/RIG query box 无 candidate id | 用 nearest-candidate by IoU，或仅 candidate scoring 模式报告 |
| CPC 与 GAIC 标签混用 | CPC pseudo score 被当 MOS | CPC 只做 pairwise，GAIC 才做 MOS/correlation |
| VLM baseline 成本失控 | API 调用慢且贵 | 先抽样 100 张，再扩展到全量 |
| scorer 不一致 | Cropper 用 VILA-R，项目用 EAT | 表中写明 scorer，做 EAT/Area/CLIP 消融 |
| 只报 pairwise 不够论文级 | 审稿人看不到标准裁剪指标 | 必须补 GAICD `Acc/SRCC/PCC` 和 `IoU/Disp` |
| image-only 主张被破坏 | 推理时读取 teacher JSON 或 candidates | 推理脚本只允许 image + checkpoint + config |
| paper-reported 与 reproduced 混淆 | 表格看似公平但设置不同 | 分表或脚注明确 |

## 12. 最终建议写法

论文方法主张建议：

```text
RIGFormer distills VLM-generated composition middle states into an image-only relation-importance graph cropper. During training, VLM/Qwen annotations provide structured supervision for entities, roles, relations and crop utility. During inference, the model uses only the input image to predict a latent composition graph and rank candidate crops through graph-conditioned utility reasoning.
```

实验主张建议：

```text
We follow standard image cropping evaluation protocols from GAICD and recent VLM-based Cropper, reporting AccK/N, rank correlations, IoU and boundary displacement error. In addition, because CPC provides comparative preferences rather than absolute MOS, we report pairwise ranking accuracy and weighted ranking accuracy.
```

不要这样写：

```text
Our method uses VLM for image cropping.
```

因为 Cropper 已经把这个点做成了主线。RIGCrop 应强调的是：

```text
VLM is only a training-time teacher; inference is image-only and graph-conditioned.
```

## 13. 参考来源

- Cropper arXiv: https://arxiv.org/abs/2408.07790
- Cropper PDF: https://arxiv.org/pdf/2408.07790
- DINOv3 ViT-B/16 model card: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
- DINOv3 collection: https://huggingface.co/collections/facebook/dinov3
- OpenAI CLIP repository: https://github.com/openai/CLIP
- CLIP ViT-B/32 HF model: https://huggingface.co/openai/clip-vit-base-patch32
- Qwen2.5-VL-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Mantis-8B-Idefics2: https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2
- GAIC official repository: https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping
- FCDB page: https://yiling-chen.github.io/flickr-cropping-dataset/
- SACD page: https://cg.cs.tsinghua.edu.cn/SACD/
- CPC / Good View Hunting project: https://www3.cs.stonybrook.edu/~cvl/projects/wei2018goods/VPN_CVPR2018s.html
- VILA paper: https://arxiv.org/abs/2303.14302
- EAT official code/resources: https://github.com/woshidandan/Image-Aesthetics-and-Quality-Assessment

