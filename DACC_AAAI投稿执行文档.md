# DACC: 面向 AAAI 投稿的具体执行文档

版本：v0.1  
日期：2026-06-15  
目标会议：AAAI-27 Main Technical Track  
目标论文方向：Direction-Aware Composition Correction for Image Cropping

## 0. AAAI 投稿约束

AAAI-27 官方时间表显示：

```text
2026-06-17  OpenReview author registration opens
2026-06-24  Paper submission site opens
2026-07-21  Abstract due, 11:59 PM UTC-12
2026-07-28  Full paper due, 11:59 PM UTC-12
2026-07-31  Supplementary material and code due, 11:59 PM UTC-12
2026-09-24  Phase 1 rejection notification
2026-10-19~25  Author feedback window
2026-11-30  Final decision
```

AAAI-27 main track 要求：

```text
技术正文最多 7 页，参考文献额外页数；
可提交技术附录、视频、代码和数据，但关键材料必须在正文中；
必须填写 reproducibility checklist；
评审关注 novelty、significance、technical soundness、clarity、responsible research 和 reproducibility。
```

因此，DACC 的执行策略必须是：

```text
1. 论文主张要非常集中；
2. 实验要足够硬；
3. 代码和数据 pipeline 要可复现；
4. 人工偏好验证要在截止前完成；
5. 不能把关键创新藏在补充材料里。
```

官方依据：

- AAAI-27 CFP: https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/
- AAAI reproducibility checklist: https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/

## 1. 论文定位

### 1.1 论文题目

主标题建议：

```text
DACC: Direction-Aware Composition Correction for Image Cropping
```

备选标题：

```text
Learning to Crop by Reasoning About Composition Corrections
Direction-Aware Image Cropping via Multimodal Teacher Distillation
Cropping Is Not Only Where: Learning Corrective Actions for Image Composition
```

### 1.2 一句话问题定义

传统图像裁剪主要学习：

```text
image -> crop box / crop score
```

DACC 学习：

```text
image -> composition issue + corrective action + top-k crop boxes + scores
```

也就是说，DACC 不只问“裁哪里”，还问：

```text
为什么要这么裁？
当前构图问题是什么？
裁剪框应该向哪移动？
应该 zoom in 还是 zoom out？
哪些主体、关键物和环境关系必须保留？
哪些区域应该被丢弃？
```

### 1.3 AAAI 叙事角度

AAAI 更看重 AI 方法、推理、学习和泛化。因此 DACC 的叙事不要写成“图像处理工具”，而要写成：

```text
一个利用多模态 teacher 产生结构化构图推理信号，
并将其蒸馏到轻量端到端裁剪模型中的 AI 方法。
```

AAAI 主线：

```text
Multimodal teacher reasoning -> structured composition supervision -> efficient student crop generator
```

## 2. 预期贡献

论文贡献必须压成 3 到 4 点。

建议最终写法：

```text
1. We introduce direction-aware image cropping, a formulation that supervises not only crop boxes and scores, but also composition issues and corrective actions.

2. We build DACC-Data, a multimodal teacher-labeled dataset containing crop-state graphs, semantic masks, action-labeled crop candidates, and ranked crop preferences.

3. We propose DACCNet, an efficient crop generator that distills multimodal teacher reasoning into real-time top-k crop prediction with auxiliary issue/action supervision.

4. Experiments on public cropping benchmarks and a human-validated DACC benchmark show improved aesthetic quality, subject preservation, relation preservation, and human preference over rule-based, ranking-based, and VLM-based baselines.
```

## 3. 最小可投稿版本和强版本

### 3.1 AAAI-27 最小可投稿版本

如果目标是 2026-07-28 前投出，建议采用压缩版：

```text
数据规模：30K-50K images
候选框：每图 30-80 个
人工验证：800-1,500 images
模型：DACCNet-Small
公开 benchmark：GAICD + FCDB + 自建 DACC-Human
baselines：GAIC-style ranker、旧规则系统、GenCrop/ProCrop/Cropper 可复现或近似 baseline、Qwen zero-shot crop
```

这版主打：

```text
新任务 + 新监督 + 明确 ablation + human preference
```

### 3.2 强版本

如果资源允许：

```text
数据规模：100K-200K images
人工验证：3K-5K images
公开 benchmark：GAICD / CPC / FCDB / FLMS / SACD
模型：DACCNet-Base + DACCNet-Small
VLM baseline：Qwen / GPT-4o / Gemini 至少两个
human study：5 annotators / image，pairwise + action correctness
```

这版可以更稳地冲 AAAI / CVPR。

## 4. 研究问题与假设

### RQ1: 方向性监督是否提升裁剪质量？

假设：

```text
加入 issue/action 辅助监督后，模型在 AccK/N、human preference、subject preservation 上优于只学 box/score 的模型。
```

验证：

```text
DACCNet full
vs w/o action loss
vs w/o issue loss
vs box-only generator
```

### RQ2: 语义 mask 和 semantic RoD 是否有用？

假设：

```text
知道被裁掉的是干扰物还是关键主体，比只看视觉 RoI/RoD 更可靠。
```

验证：

```text
full semantic scoring
vs w/o SAM/mask
vs bbox-only mask
vs visual-only ranker
```

### RQ3: VLM teacher 是否能被蒸馏成高效 student？

假设：

```text
student 不调用 Qwen/SAM 时，仍能接近或超过 VLM teacher 的偏好质量，并显著更快。
```

验证：

```text
DACCNet latency
vs Qwen zero-shot crop
vs teacher pipeline
```

### RQ4: DACC 是否比传统 crop ranking 更能保护主体关系？

假设：

```text
对于人-物、人-动物、人-环境关系图，DACC 的 relation preservation rate 显著高于 GAIC-style ranker 和规则系统。
```

验证：

```text
relation subset
metrics: Key Object Preservation, Relation Box Coverage, human pairwise preference
```

## 5. 数据构建执行方案

### 5.1 数据来源

建议组合：

```text
Public:
  GAICD / FCDB / FLMS / CPC / SACD 可用于测试和对齐

Training pool:
  Unsplash / Pexels / Flickr-compatible images / 现有业务图片
  需确认授权，避免不可公开数据拖累 reproducibility

Human validation subset:
  从训练池和 public benchmark 中抽样
```

最低规模：

```text
Train pool: 50K images
Val: 2K images
Test: 2K images
Human-validated subset: 1K images
```

强版本：

```text
Train pool: 150K images
Val: 5K images
Test: 5K images
Human-validated subset: 3K images
```

### 5.2 数据构建 pipeline

当前代码已经实现基础版：

```text
composition_dataset_builder/
```

正式论文版 pipeline：

```text
Image
  -> Qwen-VL structured understanding
  -> YOLO/Grounding/VLM bbox localization
  -> SAM mask extraction
  -> crop-state graph
  -> existing semantic rules candidates
  -> GAIC-style grid candidates
  -> direction-aware candidates
  -> synthetic bad candidates
  -> rule + mask + aesthetic + VLM scoring
  -> ranked candidates
  -> JSONL samples
```

命令模板：

```bash
python -m composition_dataset_builder.cli \
  --image-root data/images \
  --captions data/captions.json \
  --out-dir data/dacc_dataset \
  --target-aspects original,1:1,4:5,16:9 \
  --vlm qwen \
  --qwen-model qwen-vl-plus \
  --detector yolo \
  --yolo-model caption-rule-co/yolo11n.pt \
  --segmenter sam \
  --sam-checkpoint /path/to/sam_vit_h.pth \
  --aesthetic torcheat \
  --aesthetic-model caption-rule-co/TorchEAT.pth
```

### 5.3 每个样本必须保存

```json
{
  "image_path": "...",
  "target_aspect_ratio": "4:5",
  "vlm_understanding": {},
  "detections": [],
  "masks": {},
  "crop_state_graph": {},
  "candidates": [
    {
      "box": [x1, y1, x2, y2],
      "source": "direction_rule",
      "action": "preserve_relation",
      "issue": "key_object_cut",
      "features": {},
      "scores": {},
      "rank": 1
    }
  ],
  "best_crop": [x1, y1, x2, y2],
  "best_action": "preserve_relation",
  "main_issue": "key_object_cut"
}
```

### 5.4 数据质量门槛

样本级保留条件：

```text
候选数 >= 30
top1 final_score >= 3.6
至少有一个 preserve mask 或 fallback subject bbox
目标比例合法
best_crop 不越界
top1 与明显负样本分数差 >= 1.0
```

需要人工复核：

```text
top1-top2 分差 < 0.08
VLM semantic_type 与 caption router 冲突
subject coverage < 0.95 但 final_score > 4.0
key object coverage < 0.90 但 final_score > 4.0
```

## 6. 人工验证方案

AAAI 投稿必须有人类偏好验证，否则审稿人会质疑 Qwen teacher 的自循环偏差。

### 6.1 标注任务

每张图展示：

```text
原图
crop A: DACC
crop B: baseline
crop C: teacher/rule
```

标注问题：

```text
1. 哪个 crop 构图更自然？
2. 哪个 crop 更完整地保留主体？
3. 哪个 crop 更好地保留主体与关键物/环境关系？
4. 如果存在构图问题，推荐 action 是否正确？
```

### 6.2 标注规模

最低：

```text
1,000 images
3 annotators/image
pairwise comparisons: DACC vs baseline, DACC vs teacher
```

强版本：

```text
3,000 images
5 annotators/image
覆盖 semantic_type 分层采样
```

### 6.3 人工指标

```text
Human Preference Win Rate
Subject Preservation Preference
Relation Preservation Preference
Action Correctness
Inter-annotator Agreement: Fleiss' kappa / Krippendorff alpha
```

### 6.4 IRB / 伦理

如果只是让标注者评价图片构图，不收集敏感个人信息，通常属于低风险用户研究。但论文中仍需说明：

```text
标注者来源
报酬
同意流程
隐私保护
是否涉及人脸图像
数据授权
```

## 7. 模型方案：DACCNet

### 7.1 输入输出

训练输入：

```text
image
target_aspect_ratio token
optional semantic_type token
```

训练输出：

```text
K crop boxes
K crop scores
K action labels
K issue labels
```

推理输出：

```json
{
  "topk_crops": [
    {
      "box": [x1, y1, x2, y2],
      "score": 4.5,
      "action": "preserve_relation",
      "issue": "key_object_cut"
    }
  ]
}
```

### 7.2 网络结构

推荐 AAAI 版结构：

```text
Backbone:
  ConvNeXt-T / Swin-T / ViT-B

Feature Pyramid:
  multi-scale visual features

Crop Queries:
  K learnable crop queries, DETR-style

Decoder:
  cross-attention from crop queries to image features

Heads:
  box head: normalized cx, cy, w, h
  score head: crop quality
  action head: action class
  issue head: issue class
  optional coverage head: subject/relation preservation prediction
```

### 7.3 候选 ranker 辅助模型

先训练一个 ranker，降低风险：

```text
Input: image + candidate box
Output: crop score
```

Ranker 用途：

```text
1. 复现 teacher scoring；
2. 给 DACCNet 生成软标签；
3. 作为 baseline；
4. 作为 inference reranker。
```

### 7.4 损失函数

```text
L_total =
  λ_box * L_box
+ λ_iou * L_giou
+ λ_rank * L_pairwise_rank
+ λ_score * L_score_reg
+ λ_action * L_action_cls
+ λ_issue * L_issue_cls
+ λ_preserve * L_subject_relation_preserve
+ λ_div * L_topk_diversity
```

初始权重：

```text
λ_box = 2.0
λ_iou = 2.0
λ_rank = 1.0
λ_score = 0.5
λ_action = 0.5
λ_issue = 0.3
λ_preserve = 0.5
λ_div = 0.1
```

### 7.5 Matching 策略

对 top-k high-quality candidates 做 Hungarian matching：

```text
cost =
  α * box_l1
+ β * (1 - IoU)
+ γ * score_gap
+ δ * action_mismatch
```

候选正样本：

```text
final_score >= 4.0
rank <= 5
subject_coverage >= 0.95
```

负样本：

```text
negative_synthetic candidates
final_score <= 2.5
subject/key object cut
```

## 8. Baseline 设计

### 8.1 必做 baseline

```text
1. Full image / center crop baseline
2. Existing rule system
3. GAIC-style ranker
4. Crop ranker without action/issue
5. DACCNet without direction supervision
6. Qwen zero-shot crop
7. Teacher pipeline upper-bound
```

### 8.2 论文 baseline

尽量加入：

```text
GAIC
Spatial-aware Feature and Rank Consistency
GenCrop / ProCrop
Cropper / InstructCrop if code or API is available
```

如果某些代码不可复现：

```text
1. 引用其论文报告结果；
2. 在公共 benchmark 上只比较可复现方法；
3. 对 VLM 方法做统一 API prompt baseline。
```

## 9. 评价指标

### 9.1 公共裁剪指标

```text
SRCC
Acc1/5, Acc4/5, Acc1/10, Acc4/10
IoU with human crop
Boundary displacement error
Top-k recall
```

### 9.2 DACC 新指标

```text
Subject Preservation Rate:
  crop 内主体 mask 面积 / 主体 mask 总面积 >= 0.95

Key Object Preservation Rate:
  crop 内关键物 mask 面积 / 关键物 mask 总面积 >= 0.90

Relation Preservation Rate:
  crop 完整包含 relation_box 或 relation coverage >= 0.90

Semantic Discard Accuracy:
  高价值区域被保留，低价值干扰物被裁掉

Action Accuracy:
  predicted action 与 teacher/human action 一致

Issue Accuracy:
  predicted issue 与 teacher/human issue 一致

Human Preference Win Rate:
  DACC crop 被人类选中的比例

Latency:
  ms/image, CPU/GPU, batch=1
```

### 9.3 统计检验

AAAI reproducibility checklist 明确要求报告变异、显著性和实验运行次数。建议：

```text
每个模型 3 seeds
报告 mean ± std
human preference 用 bootstrap 95% CI
pairwise win rate 用 Wilcoxon signed-rank 或 sign test
```

## 10. 消融实验

必须做：

```text
DACCNet full
w/o VLM understanding
w/o SAM masks
w/o semantic RoD scoring
w/o action loss
w/o issue loss
w/o relation preservation loss
w/o synthetic negative candidates
w/o existing rule candidates
w/o top-k diversity loss
ranker-only vs generator-only vs generator+ranker
```

关键要证明：

```text
方向性监督不是装饰；
mask/semantic RoD 不是装饰；
student 不是简单复刻规则；
VLM teacher 可以被有效蒸馏。
```

## 11. 表格和图设计

### Figure 1: 任务图

展示：

```text
传统裁剪：image -> box
DACC：image -> issue/action -> top-k boxes
```

### Figure 2: Teacher pipeline

展示：

```text
Qwen -> detection -> SAM -> crop-state graph -> candidates -> scoring
```

### Figure 3: DACCNet

展示：

```text
backbone -> crop queries -> box/score/action/issue heads
```

### Figure 4: 定性结果

每行：

```text
原图
baseline crop
DACC crop
action/reason
```

重点挑：

```text
主体贴边
人和物体
动作留白
干扰物去除
重要环境保留
```

### Table 1: Public benchmark

```text
Method | SRCC | Acc5 | Acc10 | IoU | Latency
```

### Table 2: DACC-Human

```text
Method | Human Win | Subject Preserve | Relation Preserve | Action Acc | Latency
```

### Table 3: Ablation

```text
Variant | Human Win | Relation Preserve | Acc5 | Action Acc
```

### Table 4: Teacher vs Student

```text
Method | Quality | Latency | Cost | Needs VLM/SAM at inference?
```

## 12. 执行时间表

当前日期：2026-06-15。距离 AAAI-27 abstract deadline 约 5 周，距离 full paper deadline 约 6 周。

### Week 1: 2026-06-15 到 2026-06-21

目标：

```text
冻结论文主张；
跑通 5K 数据构建；
训练 ranker baseline；
确定 public benchmark 和 baseline 可运行性。
```

交付：

```text
5K DACC pilot JSONL
100 张可视化人工检查
ranker v0
paper skeleton
```

Go/No-Go：

```text
如果 teacher labels 视觉抽查通过率 < 70%，先修数据 pipeline，不训练 student。
```

### Week 2: 2026-06-22 到 2026-06-28

目标：

```text
构建 30K-50K 数据；
训练 DACCNet-Small v0；
完成旧规则、GAIC-style、Qwen zero-shot baselines。
```

交付：

```text
DACC-Data v0
DACCNet-Small v0
baseline result table v0
```

Go/No-Go：

```text
DACCNet 必须在至少 2 个核心指标超过旧规则或 GAIC-style ranker。
```

### Week 3: 2026-06-29 到 2026-07-05

目标：

```text
加入 action/issue/mask ablation；
开始人工偏好标注；
完成 public benchmark 初版。
```

交付：

```text
ablation v0
human study interface / CSV
public benchmark table v0
```

Go/No-Go：

```text
w/o action loss 与 full 差距必须可见；
否则论文主张要转向 semantic RoD 或 teacher distillation。
```

### Week 4: 2026-07-06 到 2026-07-12

目标：

```text
完成 1K human validation；
修模型；
完成图表；
写完 methods 和 experiments。
```

交付：

```text
human preference table
main result table
figure 1-4 draft
paper draft v0
```

Go/No-Go：

```text
Human preference win rate 至少 > 55%，且 95% CI 下界最好 > 50%。
```

### Week 5: 2026-07-13 到 2026-07-20

目标：

```text
完成论文初稿；
补统计显著性；
补 reproducibility checklist；
准备 abstract。
```

交付：

```text
AAAI abstract
full paper v1
appendix v0
code/data release plan
```

### 2026-07-21

提交 abstract。

### Week 6: 2026-07-22 到 2026-07-28

目标：

```text
压缩 7 页；
最终实验补漏；
内部 review；
提交全文。
```

交付：

```text
camera submission PDF
supplement checklist
匿名代码包
```

### 2026-07-31

提交 supplementary material 和 code。

## 13. 论文结构

AAAI 7 页正文建议：

```text
1. Introduction                  1.0 page
2. Related Work                  0.8 page
3. Direction-Aware Cropping      0.7 page
4. DACC-Data Construction        1.0 page
5. DACCNet                       1.0 page
6. Experiments                   1.8 pages
7. Conclusion                    0.2 page
```

正文必须放：

```text
任务定义
核心 pipeline
模型结构
主要结果
关键消融
human study
reproducibility 要点
```

补充材料放：

```text
完整 prompt
更多定性图
数据 schema
更多 ablation
人工标注细节
超参数
失败案例
```

## 14. Abstract 初稿

```text
Image cropping is commonly formulated as crop-box regression or candidate ranking. However, high-quality cropping often requires directional composition correction: moving the frame, zooming in or out, preserving subject-object relations, and discarding distracting regions. We introduce Direction-Aware Composition Correction (DACC), a new formulation that supervises not only where to crop, but also what composition issue is present and which corrective action should be taken. To obtain such supervision at scale, we construct a multimodal teacher pipeline that combines vision-language reasoning, semantic segmentation, rule-based candidate generation, and preference scoring to produce crop-state graphs with action-labeled ranked candidates. We further propose DACCNet, an efficient crop generator that distills the teacher's reasoning into top-k crop predictions with auxiliary issue and action supervision. Experiments on public cropping benchmarks and a human-validated DACC benchmark show that DACCNet improves crop quality, subject preservation, relation preservation, and human preference over rule-based, ranking-based, and VLM-based baselines, while requiring no multimodal teacher at inference time.
```

## 15. Reproducibility Checklist 准备

AAAI checklist 需要提前准备：

```text
1. 方法伪代码
2. 数据来源和授权说明
3. 数据预处理代码
4. 模型代码
5. 所有 hyperparameter 搜索范围
6. 最终 hyperparameters
7. random seeds
8. GPU/CPU/内存/系统/库版本
9. 评价指标定义
10. 每个结果的运行次数
11. mean/std 或 CI
12. 统计显著性检验
13. 数据和代码公开计划
```

当前项目应补：

```text
configs/dacc_small.yaml
configs/dacc_base.yaml
scripts/build_dacc_data.sh
scripts/train_ranker.sh
scripts/train_daccnet.sh
scripts/eval_public_benchmarks.sh
scripts/eval_human_subset.sh
```

## 16. 风险清单和回退策略

### 风险 1: 被认为只是 pipeline 拼装

回退：

```text
强调 student distillation；
加 action/issue loss 的强消融；
证明 inference 不需要 Qwen/SAM；
突出 semantic RoD 和 direction-aware task。
```

### 风险 2: VLM teacher 标签不稳定

回退：

```text
只让 Qwen 输出高层语义；
bbox/mask 依赖 YOLO/SAM；
action 标签用规则校验；
人工验证 top subset。
```

### 风险 3: human preference 不显著

回退：

```text
改主张为 relation preservation / subject preservation；
强调客观语义指标；
减少审美泛化 claim。
```

### 风险 4: public benchmark 提升不明显

回退：

```text
把贡献定位到新任务和 DACC-Human benchmark；
公开 benchmark 作为兼容性测试；
主结果放 relation/action subset。
```

### 风险 5: 时间不够

回退：

```text
先投 AAAI workshop / arXiv；
主会版保留完整 human study 和 stronger baselines；
或者转 AAAI-28/CVPR。
```

## 17. 最低录用门槛

如果要有 AAAI main track 竞争力，建议至少达到：

```text
Human Preference Win Rate:
  DACC vs best non-VLM baseline >= 58%

Subject Preservation:
  +5% absolute over GAIC-style / rule baseline

Relation Preservation:
  +8% absolute on relation subset

Action Ablation:
  full model > w/o action by >= 2% human win or >= 3% relation preservation

Latency:
  student at least 10x faster than Qwen/SAM teacher pipeline

Reproducibility:
  code/data/splits/configs can be released or anonymized
```

如果这些达不到，主会风险会很高。

## 18. 当前工程任务列表

### 数据

```text
[ ] 准备 50K 授权图片
[x] 跑通 composition_dataset_builder MVP 代码
[ ] 接入 Qwen 输出 vlm_understanding
[ ] 接入 YOLO / Grounding
[ ] 接入 SAM
[ ] 生成 DACC-Data v0
[ ] 抽查 500 张可视化
[ ] 生成人工验证子集
```

### 模型

```text
[x] 实现 candidate ranker baseline
[x] 实现 DACCNet-Small baseline
[x] 实现 action/issue heads
[ ] 实现 Hungarian top-k matching
[ ] 实现 pairwise/listwise rank loss
[ ] 实现 ablation switches
```

### 实验

```text
[ ] GAIC-style baseline
[ ] rule baseline
[ ] Qwen zero-shot baseline
[ ] teacher upper bound
[ ] public benchmark eval
[ ] DACC-Human eval
[ ] ablation
[ ] latency test
```

### 论文

```text
[ ] figure 1 task definition
[ ] figure 2 teacher pipeline
[ ] figure 3 DACCNet
[ ] figure 4 qualitative results
[ ] table 1 public benchmark
[ ] table 2 DACC-Human
[ ] table 3 ablation
[ ] appendix prompt/schema
[ ] reproducibility checklist
```

## 19. 最终建议

如果目标是 AAAI-27，主线要非常明确：

```text
DACC 不是一个裁剪工程系统；
DACC 是一种方向感知的构图修正学习框架。
```

最终论文的胜负点不是 pipeline 多复杂，而是：

```text
1. action/issue 监督是否真的提升结果；
2. semantic mask / semantic RoD 是否真的保护了主体关系；
3. student 是否能高效替代 VLM teacher；
4. 人类是否更喜欢 DACC 的裁剪。
```

只要这四点站住，AAAI 是可以冲的。

## 20. 当前可执行代码入口

当前已经新增 DACC 训练与评估代码：

```text
dacc/
scripts/
configs/
```

一键 smoke test：

```bash
bash scripts/run_smoke_tests.sh
```

真实数据训练：

```bash
python scripts/train_ranker.py --config configs/ranker_small.yaml
python scripts/train_daccnet.py --config configs/daccnet_small.yaml
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

详细说明见：

```text
DACC代码执行手册.md
```
