# RIG-Crop 中间态驱动 Image-only 裁剪 Student 执行方案

日期：2026-06-23

本文档使用 `academic-research-suite` 的 deep-research + experiment-plan 方式整理。目标不是再做一个普通 crop ranker，而是把已经生成的 Qwen / VLM 中间态直接转成可训练监督，设计一个推理时只输入图片、但训练时学习结构化构图推理的 student。

---

## 0. Material Passport

- Origin Skill: academic-research-suite
- Workflow: deep-research + experiment-agent plan
- Verification Status: PARTIALLY VERIFIED
- Scope: image cropping, composition-aware cropping, VLM-distilled middle-state supervision
- Local project root: `/Users/hmx/XMU/image_cut`
- Current available data path pattern:
  - CPC raw DACC: `data/cpc_dacc/metadata/{train,val}.jsonl`
  - CPC Qwen middle state: `data/cpc_semantic_qwen/metadata/{train,val}.jsonl`
  - GAICD DACC / Qwen middle state: follow existing `GAICD_VLM中间态生成说明.md`
- Key constraint: inference input must be one image only. VLM, detector, SAM, Qwen JSON, candidate labels are training/offline assets only.

---

## 1. 一句话方案

推荐方法名：

```text
RIG-Crop: Relation-Importance Guided Image Cropping
```

核心主张：

```text
Use VLM-generated composition middle states as structured training supervision
for entity importance, inter-entity relation preservation, and crop policy.
At inference, the student receives only one image and internally predicts a
latent composition graph to rank or generate top-k crops.
```

中文表述：

```text
用 Qwen/VLM 离线生成“主体-重要性-关系-裁剪策略”中间态，
把它转成结构化监督训练一个 image-only student。
推理时 student 不读取 caption、不调用 Qwen、不读取中间态 JSON，
只输入一张图片，输出 top-k 裁剪框和分数。
```

这比“用 VLM 生成数据再训练模型”更有创新性，因为核心不是伪标签，而是结构化构图推理蒸馏：

- 谁重要：entity importance
- 谁必须一起保留：relation preservation
- 哪些关系被 crop 边界破坏：crop-conditioned relation utility
- 哪些区域可丢弃：distractor penalty
- 为什么某个 crop 更好：importance/relation/policy 共同解释

---

## 2. 文献调研与定位

### 2.1 搜索范围

搜索目标：

- automatic image cropping
- photo composition learning
- candidate crop ranking
- pairwise crop preference
- relation-aware cropping
- composition-aware cropping
- VLM/MLLM image cropping
- weakly supervised/professional composition cropping

优先来源：

- CVF Open Access: CVPR / ICCV
- AAAI OJS
- arXiv only when conference page unavailable or for extended version
- official project page / dataset page

### 2.2 Literature Matrix

| 方法 | Venue | 核心机制 | 对本项目的启发 | 与 RIG-Crop 的区别 |
|---|---:|---|---|---|
| A2-RL | CVPR 2018 | 把裁剪建模为序列决策，用 aesthetic reward 学习裁剪动作 | 裁剪可以是 action/policy 问题，不一定只是回归框 | 没有 VLM 结构化中间态，也没有显式 entity relation |
| CPC / Good View Hunting | CVPR 2018 | 大规模 comparative view pairs，训练 view ranking/proposal | CPC 适合 pairwise ranking，不要强行当 MOS 回归 | CPC 只有偏好，没有解释谁重要、关系为何保留 |
| GAIC / GAICD | CVPR 2019 | Grid anchor candidate scoring，每张图少量候选框全标注 MOS | internal anchor scoring 是稳定第一版 student 的好选择 | GAIC 的 RoI/RoD 不是语义化关系图 |
| Composition + Saliency Aesthetic Map | AAAI 2020 | 组合 saliency 与 composition-aware aesthetic map | 显著性和构图位置都重要 | importance 仍偏 saliency，不是 VLM role-aware importance |
| Mutual Relations | CVPR 2020 | candidate region graph，候选之间传播关系 | 不能再声称“我们首次建图” | 它建的是候选 region 关系，不是 entity-role-relation-policy |
| Composing Photos Like a Photographer | CVPR 2021 | Key Composition Map 显式编码构图规则 | 顶会认可显式构图知识 | KCM 不是 instance-level subject importance/relation |
| TransView | ICCV 2021 | 建模 crop 内、外、跨边界 visual words 的 attraction/repulsion | 关系必须是 crop-conditioned，不能只看全图静态关系 | 关系是隐式 visual word 依赖，不是 Qwen typed relation |
| Rethinking Image Cropping | CVPR 2022 | set prediction，多 learnable anchors 生成多样 crop | 最终应输出 top-k crops，而不是唯一框 | 解决多样性，不解释主体/关系取舍 |
| Spatial-aware Feature + Rank Consistency | CVPR 2023 | crop 与 aesthetic elements 的空间关系 + rank consistency | crop 与元素的相对空间是强监督 | aesthetic elements 可升级为 VLM entity graph |
| S2CNet | AAAI 2024 | spatial-semantic adaptive attention graph，UGCrop5K | 最接近“多主体关系”方向，必须重点对比 | S2CNet 是隐式 visual node graph；RIG-Crop 是 VLM-distilled explicit graph supervision |
| GenCrop | AAAI 2024 | professional photos + diffusion outpainting 构造 weakly-supervised pairs | 可作为大规模预训练数据启发 | 主体感知为主，不提供 relation/importance/policy graph |
| Cropper | CVPR 2025 | VLM in-context learning + iterative refinement | VLM 已经能做 crop reasoning，可作为 teacher/upper bound | Cropper 推理时依赖大 VLM；RIG-Crop 推理是轻量 image-only |
| PICD | CVPR 2025 | composition category dataset/benchmark | 可作为 composition understanding 辅助评估 | PICD 不直接给裁剪框偏好 |
| ProCrop | AAAI 2026 | professional composition retrieval + 242K weak labels | 大规模弱监督 crop 数据趋势明确 | ProCrop 学专业构图检索/弱标签；RIG-Crop 学语义取舍和关系保留 |

### 2.3 关键证据综合

#### 结论 A：仅做 crop ranking baseline 不够新

CPC 已经证明 comparative view pairs 对构图学习有效；GAIC 已经把裁剪变成 grid-anchor scoring，并用 dense candidate MOS 解决单一框不可靠的问题。你的模型如果只是把 CPC/GAIC 混合起来训一个 ranker，属于合理工程，但创新弱。

#### 结论 B：仅说“多主体关系”也不够新

Mutual Relations、TransView、S2CNet 都已经证明关系建模对裁剪有帮助。尤其 S2CNet 已经面向 UGC 的多物体复杂背景，使用 spatial-semantic graph 和 message passing。因此论文不能写成“我们考虑关系”。必须写成：

```text
Existing relation-aware croppers learn implicit visual or candidate relations.
We distill explicit VLM composition reasoning into trainable supervision:
entity role, entity importance, typed relation, crop boundary policy,
and relation-preservation utility.
```

#### 结论 C：VLM cropping 已经出现，所以不能把“用 VLM”本身当创新

Cropper 显示 VLM 可以通过 in-context learning 做 cropping，并能适配 free-form、subject-aware、aspect-ratio-aware cropping。你的差异必须是：

```text
VLM is not used at inference. It is a teacher that produces structured
middle states. The deployable model is a compact image-only cropper.
```

#### 结论 D：中间态最有价值的是结构，不是 caption

caption 种类太散，小模型很难学，而且容易变成开放类别识别。更合理的是把 Qwen 输出转成低维结构化监督：

- role: main_subject / key_object / background / distractor
- importance: 0-1
- bbox_norm: [x1, y1, x2, y2]
- relation_to_subject: relation text normalized to small policy classes
- preserve / optional_preserve / avoid_cutting
- leave_space_direction / preferred_subject_position / suggested_actions

---

## 3. 你的当前数据应该怎么用

### 3.1 已有输入

CPC Qwen 版本每行应该包含：

```json
{
  "image_path": "...",
  "candidates": [
    {"candidate_id": "cpc_010", "box": [114, 0, 1024, 683], "scores": {...}}
  ],
  "pairwise_preferences": [
    {"winner": "cpc_010", "loser": "cpc_003", "weight": 0.31}
  ],
  "composition_middle_state": {
    "main_subject": {...},
    "key_objects": [...],
    "important_background": [...],
    "distractors": [...],
    "composition_intent": {...}
  }
}
```

Qwen prompt 已经要求：

```text
main_subject: name/category/description/importance/bbox_norm
key_objects: name/category/relation_to_subject/importance/bbox_norm
important_background: name/category/importance/bbox_norm
distractors: name/category/importance/location/bbox_norm
composition_intent:
  preserve, optional_preserve, avoid_cutting, leave_space_direction,
  preferred_subject_position, initial_issue, suggested_actions
```

### 3.2 不建议直接训练的字段

不要直接训练 student 预测：

- 任意 caption 文本
- 任意 object category 文本
- 任意 relation_to_subject 原始字符串

原因：

- 类别开放，样本量不够，长尾严重。
- caption 监督和 crop 目标弱相关。
- 小模型会被迫学习“全类别检测 + captioning”，难度超过裁剪本身。

### 3.3 建议直接训练的字段

应该训练：

```text
node box: bbox_norm
node role: main/key/background/distractor
node importance: 0-1
node confidence: 可选
relation policy: preserve / optional / avoid_cutting / distractor_exclusion / none
crop-conditioned utility: 每个候选 crop 对节点和关系的保留程度
```

这可以把“开放文本”压缩成稳定的构图结构。

---

## 4. 方法设计：RIG-Crop

### 4.1 总体结构

```text
image
  -> visual backbone
  -> latent entity queries
       -> node boxes, roles, importance
       -> relation matrix
  -> internal crop candidates / dataset crop candidates
       -> crop-conditioned graph utility
       -> crop score
  -> top-k crops
```

训练时：

```text
image + dataset candidates + CPC/GAIC crop labels + Qwen middle-state targets
```

推理时：

```text
image only
```

模型内部自己生成 candidate crops，例如 GAIC-style anchors 或 learnable crop queries。

### 4.2 两阶段实现路线

#### Version 1：Anchor-ranking RIG-Crop

这是最稳、最快能做出结果的版本。

```text
image -> deterministic GAIC-style anchors -> score each anchor
```

优点：

- 和 GAIC/CPC 训练方式兼容。
- 容易对比现有 ranker。
- 中间态可以作为辅助监督，不改变主任务定义。

缺点：

- 推理输出受 anchor 空间限制。
- 创新主要在 latent graph + utility loss。

#### Version 2：Query-generation RIG-Crop

在 Version 1 跑通后升级：

```text
image -> K crop queries -> top-k crop boxes + scores
```

优点：

- 更像 Rethinking Image Cropping 的 set prediction 方向。
- 输出更灵活。

缺点：

- 实现难度更高。
- 需要更仔细的 matching 和多样性约束。

AAAI 第一版建议先做 Version 1，补一个 Version 2 作为扩展实验或未来工作。

---

## 5. 中间态标准化：从 Qwen JSON 到 Graph Targets

### 5.1 输出目标文件

新增脚本：

```text
scripts/build_middle_state_targets.py
```

输入：

```text
data/cpc_semantic_qwen/metadata/train.jsonl
data/cpc_semantic_qwen/metadata/val.jsonl
```

输出：

```text
data/cpc_rig/metadata/train.jsonl
data/cpc_rig/metadata/val.jsonl
```

每行新增：

```json
{
  "rig_targets": {
    "nodes": [...],
    "relations": [...],
    "candidate_utilities": {...},
    "graph_quality_flags": {...}
  }
}
```

### 5.2 Node 结构

最多保留 `M=8` 个节点：

```json
{
  "node_id": 0,
  "source": "main_subject",
  "role": "main_subject",
  "bbox_norm": [0.18, 0.22, 0.74, 0.91],
  "importance": 1.0,
  "valid_box": true
}
```

role 映射：

```text
main_subject -> 0
key_object -> 1
important_background -> 2
distractor -> 3
padding -> 4
```

### 5.3 Relation 结构

关系不要使用原始开放字符串，统一成 policy class：

```text
preserve_relation
optional_preserve
avoid_cutting
leave_space
distractor_exclusion
none
```

构造规则：

- main_subject 与 key_objects：
  - 如果对象在 `preserve` 或 `avoid_cutting` 中出现，设为 `preserve_relation`
  - 否则若 importance >= 0.6，设为 `optional_preserve`
- main_subject 与 important_background：
  - 设为 `leave_space` 或 `optional_preserve`
- main_subject 与 distractors：
  - 设为 `distractor_exclusion`

矩阵形式：

```text
R_policy: [M, M] int
R_weight: [M, M] float
```

relation weight：

```text
w_ij = sqrt(importance_i * importance_j)
```

### 5.4 Candidate Utility

对每个 candidate crop `c`，从 teacher graph 计算一个结构化 utility：

```text
U(c) =
  sum_i importance_i * coverage(node_i, c)
  - lambda_cut * sum_i importance_i * boundary_cut(node_i, c)
  + lambda_rel * sum_ij relation_weight_ij * relation_preserved(i, j, c)
  - lambda_dist * sum_d distractor_importance_d * coverage(distractor_d, c)
  + lambda_pos * subject_position_score(main_subject, c)
```

解释：

- `coverage(node_i, c)`: 节点框落入 crop 的面积比例。
- `boundary_cut(node_i, c)`: crop 是否切断高重要节点。
- `relation_preserved(i, j, c)`: 两个相关节点是否同时被保留，且没有被边界切断。
- `distractor coverage`: 干扰物被裁进来要扣分。
- `subject_position_score`: 主体是否在中心/三分线/留白方向合适。

这个 `U(c)` 不替代 CPC/GAIC 的人工监督，而是作为辅助蒸馏目标。

### 5.5 质量过滤

必须过滤或降权：

```text
main_subject 无 bbox_norm -> 降权 node bbox loss，但保留 importance/role
所有节点无 bbox_norm -> 样本只用于 crop rank，不用于 graph loss
Qwen failed / JSON malformed -> 剔除或退回 heuristic
节点 bbox 超界 -> clip 到 [0,1] 并标记
```

---

## 6. Student 模型结构

### 6.1 输入输出定义

训练输入：

```text
image: RGB image
candidates: dataset candidates or generated anchors
crop labels:
  CPC pairwise_preferences
  GAIC MOS / MOS-derived pairs
middle state:
  rig_targets.nodes
  rig_targets.relations
  rig_targets.candidate_utilities
```

训练输出：

```text
score(crop)
pred_nodes
pred_relations
pred_candidate_utility
optional action logits
```

推理输入：

```text
image only
```

推理输出：

```text
top-k crop boxes
top-k crop scores
optional predicted latent graph for visualization
```

### 6.2 Backbone

第一版建议：

```text
ConvNeXt-T / Swin-T / DINOv2-S frozen or partially frozen
```

如果服务器资源有限：

```text
ResNet50 or current CropRanker backbone
```

投稿版本建议至少用一个强 backbone，否则 reviewer 会认为提升来自 backbone 不足或 baseline 弱。

### 6.3 Latent Entity Query Head

使用 DETR-like `M` 个 entity queries：

```text
query embeddings Q_e: [M, D]
cross-attend to image tokens
output entity tokens E: [M, D]
```

预测：

```text
bbox_head(E_i) -> [x1,y1,x2,y2]
role_head(E_i) -> role logits
importance_head(E_i) -> scalar 0-1
valid_head(E_i) -> objectness/validity
```

训练 matching：

```text
Hungarian matching between predicted nodes and teacher nodes
cost = L1_bbox + CE_role + MSE_importance
```

### 6.4 Relation Head

对 entity tokens 做 pairwise relation：

```text
r_ij = MLP([E_i, E_j, E_i * E_j, |E_i - E_j|])
```

输出：

```text
relation_policy_logits: [M, M, C]
relation_weight: [M, M]
```

监督：

```text
L_relation_policy = CE over valid teacher relation entries
L_relation_weight = SmoothL1(pred_weight, teacher_weight)
```

### 6.5 Crop Candidate Scorer

候选来源：

训练：

```text
CPC candidates / GAIC candidates
```

推理：

```text
internal GAIC-style anchors
```

对每个 candidate crop `c`：

```text
crop_feat = RoIAlign(image_tokens, c) or crop resize branch
box_feat = [x1,y1,x2,y2,w,h,area,aspect]
graph_feat(c) = graph utility features computed from predicted nodes/relations
score(c) = MLP([global_feat, crop_feat, box_feat, graph_feat])
```

`graph_feat(c)` 包含：

```text
pred_entity_coverage_sum
pred_relation_preservation_sum
pred_distractor_inclusion_sum
pred_boundary_cut_penalty
pred_subject_position_score
```

这一步是创新核心：crop score 不是只看 crop 视觉特征，而是看 predicted latent graph 在该 crop 下是否被合理保留。

---

## 7. Loss 设计

总损失：

```text
L =
  L_crop
  + lambda_node * L_node
  + lambda_rel * L_relation
  + lambda_util * L_utility
  + lambda_action * L_action
```

### 7.1 Crop Loss

CPC：

```text
L_cpc = softplus(-(score(winner) - score(loser))) * pair_weight
```

GAIC：

```text
L_gaic_reg = SmoothL1(score(c), normalized_MOS(c))
L_gaic_pair = softplus(-(score(high_MOS) - score(low_MOS)))
```

混训：

```text
L_crop = L_cpc + alpha * L_gaic_pair + beta * L_gaic_reg
```

### 7.2 Node Loss

```text
L_node =
  L1(pred_bbox, teacher_bbox)
  + GIoU(pred_bbox, teacher_bbox)
  + CE(pred_role, teacher_role)
  + SmoothL1(pred_importance, teacher_importance)
  + BCE(pred_valid, teacher_valid)
```

如果 Qwen 节点没有 bbox：

```text
disable bbox/GIoU loss for that node
keep role/importance loss if valid
```

### 7.3 Relation Loss

```text
L_relation =
  CE(pred_relation_policy_ij, teacher_policy_ij)
  + SmoothL1(pred_relation_weight_ij, teacher_weight_ij)
```

只在 valid node pair 上计算。

### 7.4 Candidate Utility Distillation

Teacher utility:

```text
U_teacher(c) = graph-derived utility from Qwen nodes/relations
```

Student utility:

```text
U_student(c) = graph utility computed from predicted nodes/relations
```

Loss：

```text
L_utility = SmoothL1(U_student(c), U_teacher(c))
```

或者对同一图内 pairwise utility：

```text
L_utility_pair = softplus(-(U_student(c_good) - U_student(c_bad)))
```

建议第一版用 pairwise utility，更稳：

```text
if U_teacher(c_i) > U_teacher(c_j) + margin:
  U_student(c_i) should be > U_student(c_j)
```

### 7.5 Action / Policy Loss

从 `composition_intent.suggested_actions` 取多标签：

```text
L_action = BCEWithLogits(action_logits, action_multi_hot)
```

这个 loss 权重不要太大，建议：

```text
lambda_action = 0.05
```

---

## 8. 具体实现任务

### 8.1 新增数据预处理脚本

新增：

```text
scripts/build_middle_state_targets.py
```

功能：

```text
input:  qwen enriched JSONL
output: JSONL with rig_targets
```

命令：

```bash
python scripts/build_middle_state_targets.py \
  --input-jsonl data/cpc_semantic_qwen/metadata/train.jsonl \
  --out-jsonl data/cpc_rig/metadata/train.jsonl \
  --max-nodes 8 \
  --overwrite

python scripts/build_middle_state_targets.py \
  --input-jsonl data/cpc_semantic_qwen/metadata/val.jsonl \
  --out-jsonl data/cpc_rig/metadata/val.jsonl \
  --max-nodes 8 \
  --overwrite
```

### 8.2 新增 Dataset

新增或扩展：

```text
dacc/data.py
  RIGCropPairwiseDataset
  RIGCropImageDataset
```

输出 batch：

```python
{
  "image": Tensor[B,3,H,W],
  "winner_crop": Tensor[B,3,h,w],
  "loser_crop": Tensor[B,3,h,w],
  "winner_box_feat": Tensor[B,8],
  "loser_box_feat": Tensor[B,8],
  "rig_nodes": Tensor[B,M,...],
  "rig_relations": Tensor[B,M,M],
  "rig_candidate_utility": Tensor[B,N],
  "weight": Tensor[B]
}
```

### 8.3 新增模型

新增：

```text
dacc/models/rig_crop.py
```

类：

```python
class RIGCrop(nn.Module):
    def forward_train(self, image, candidate_boxes, rig_targets=None):
        ...

    def forward_infer(self, image, topk=5, aspect_ratio=None):
        ...
```

训练 forward 返回：

```python
{
  "crop_scores": ...,
  "pred_nodes": ...,
  "pred_relations": ...,
  "pred_utilities": ...,
  "loss_inputs": ...
}
```

### 8.4 新增训练脚本

新增：

```text
scripts/train_rig_crop.py
```

第一版只做 CPC：

```bash
python scripts/train_rig_crop.py \
  --config configs/rig_crop_cpc.yaml
```

第二版混训：

```bash
python scripts/train_rig_crop.py \
  --config configs/rig_crop_cpc_gaic.yaml
```

### 8.5 新增评估脚本

新增：

```text
scripts/eval_rig_crop.py
```

评估内容：

- CPC pairwise accuracy
- weighted pairwise accuracy
- score margin
- top-1 crop visualization
- graph target quality:
  - node bbox IoU
  - role accuracy
  - importance MAE
  - relation policy accuracy

---

## 9. 实验设计

### 9.1 Baselines

必须有：

| ID | 方法 | 数据 | 中间态 | 目的 |
|---|---|---|---|---|
| B1 | current CropRanker | CPC | no | CPC pairwise baseline |
| B2 | current CropRanker | CPC + GAIC | no | 混训基础线 |
| B3 | S2C-style implicit graph variant | CPC | no explicit Qwen | 证明不是普通 graph 就够 |
| B4 | RIG-Crop w/o relation | CPC Qwen | node importance only | 消融 relation |
| B5 | RIG-Crop w/o importance | CPC Qwen | relation only | 消融 importance |
| B6 | RIG-Crop w/o utility | CPC Qwen | graph aux only | 消融 crop-conditioned utility |
| Ours | RIG-Crop full | CPC Qwen + GAIC Qwen | full | 主结果 |

建议如果时间有限，先跑：

```text
B1, B4, B6, Ours
```

### 9.2 主指标

CPC：

```text
pairwise_acc
weighted_pairwise_acc
mean_score_margin
```

GAICD：

```text
Spearman rank correlation
Kendall tau
top-1 MOS
top-k recall of high-MOS candidates
```

可解释性：

```text
node importance MAE
node bbox mIoU against Qwen bbox
relation policy accuracy
utility rank consistency
```

推理效率：

```text
FPS
params
FLOPs
no VLM inference cost
```

### 9.3 消融重点

最能支撑创新的消融：

```text
RIG-Crop full
minus relation loss
minus importance loss
minus utility distillation
minus node bbox supervision
replace Qwen with heuristic middle state
replace crop-conditioned relation with static relation graph
```

最后一个很重要，因为它直接证明：

```text
关系不是全图静态存在，而是会被具体 crop 边界保留或破坏。
```

### 9.4 可视化

每张图输出：

- 原图
- top-5 predicted crops
- predicted entity nodes
- predicted importance heat/box
- predicted relation edges
- best crop 与 relation preservation 的解释

示例解释模板：

```text
The selected crop preserves the main subject and the bicycle relation,
while excluding a low-importance distractor on the right.
```

注意：论文中不要过度声称“真实解释”，应该写成：

```text
model-internal explanation aligned with VLM-distilled composition state
```

---

## 10. 数据路线

### 10.1 第一阶段数据

必须使用：

```text
CPC: 约 10.8K images, pairwise preference
GAICD: 约 1K images, dense candidate MOS
```

CPC 的作用：

- 大量 pairwise crop preference
- 适合训练 ranker

GAICD 的作用：

- dense candidate MOS
- 适合校准 crop score

Qwen middle state 的作用：

- 训练 latent graph
- 训练 importance/relation/policy
- 训练 candidate utility

### 10.2 第二阶段可扩展数据

可选：

```text
UGCrop5K
ProCrop 242K weak labels
GenCrop cropped-uncropped pairs
PICD composition categories
```

建议优先级：

```text
1. CPC + GAICD + Qwen
2. Add UGCrop5K if available
3. Add ProCrop for pretraining crop score
4. Add PICD only for auxiliary composition representation, not main crop supervision
```

不要一开始就把所有数据混进去。先证明中间态有效，再扩大数据。

---

## 11. 训练计划

### Stage 0：质量审查

检查 Qwen 输出：

```bash
python - <<'PY'
import json
path = "data/cpc_semantic_qwen/metadata/train.jsonl"
bad = total = has_main_box = has_key_box = 0
for line in open(path, encoding="utf-8"):
    total += 1
    r = json.loads(line)
    ms = r.get("composition_middle_state", {})
    main = ms.get("main_subject", {})
    if isinstance(main, dict) and len(main.get("bbox_norm", []) or []) == 4:
        has_main_box += 1
    for obj in ms.get("key_objects", []) or []:
        if isinstance(obj, dict) and len(obj.get("bbox_norm", []) or []) == 4:
            has_key_box += 1
            break
print("total", total, "main_box_rate", has_main_box / max(total,1), "key_box_image_rate", has_key_box / max(total,1))
PY
```

如果 main bbox rate < 0.6，先改 Qwen prompt 或用 fallback grounding，不要急着训 graph head。

### Stage 1：构建 RIG targets

```bash
python scripts/build_middle_state_targets.py \
  --input-jsonl data/cpc_semantic_qwen/metadata/train.jsonl \
  --out-jsonl data/cpc_rig/metadata/train.jsonl \
  --max-nodes 8 \
  --overwrite

python scripts/build_middle_state_targets.py \
  --input-jsonl data/cpc_semantic_qwen/metadata/val.jsonl \
  --out-jsonl data/cpc_rig/metadata/val.jsonl \
  --max-nodes 8 \
  --overwrite
```

### Stage 2：Graph pretraining

只训练：

```text
node box / role / importance / relation
```

命令建议：

```bash
python scripts/train_rig_crop.py \
  --config configs/rig_crop_graph_pretrain_cpc.yaml
```

### Stage 3：Joint crop training

训练：

```text
crop rank + graph + utility
```

命令：

```bash
python scripts/train_rig_crop.py \
  --config configs/rig_crop_cpc_joint.yaml
```

### Stage 4：GAICD 混训

命令：

```bash
python scripts/train_rig_crop.py \
  --config configs/rig_crop_cpc_gaic_joint.yaml
```

### Stage 5：推理测试

```bash
python scripts/predict_rig_crop.py \
  --checkpoint runs/rig_crop_cpc_gaic_joint/best.pt \
  --image examples/test.jpg \
  --out runs/rig_crop_demo/test_vis.jpg \
  --topk 5
```

---

## 12. 推荐配置

### 12.1 `configs/rig_crop_cpc_joint.yaml`

```yaml
seed: 42
device: auto
output_dir: runs/rig_crop_cpc_joint

batch_size: 16
num_workers: 4
epochs: 30
lr: 0.0001
weight_decay: 0.0001

train_dataset:
  jsonl_path: data/cpc_rig/metadata/train.jsonl
  image_size: 384
  crop_size: 224
  max_nodes: 8
  max_pairs_per_record: 128

val_dataset:
  jsonl_path: data/cpc_rig/metadata/val.jsonl
  image_size: 384
  crop_size: 224
  max_nodes: 8
  max_pairs_per_record: 128

model:
  backbone: convnext_tiny
  feat_dim: 256
  num_entity_queries: 8
  num_relation_classes: 6
  num_actions: 15
  anchor_generator: gaic_grid

loss:
  crop_pair: 1.0
  node: 0.3
  relation: 0.2
  utility: 0.3
  action: 0.05
```

---

## 13. 论文创新点写法

### 13.1 可以主张的创新

```text
1. We introduce a VLM-distilled composition middle-state representation
   for image cropping, including entity importance, typed preservation policy,
   and crop-conditioned relation utility.

2. We propose an image-only student cropper that learns to predict a latent
   composition graph during training and uses it to rank candidate crops at
   inference without calling any VLM.

3. We design a crop-conditioned relation-preservation utility that converts
   VLM reasoning into differentiable crop ranking supervision.
```

### 13.2 不建议主张

不要写：

```text
We are the first to use relations for cropping.
We are the first to use VLM for cropping.
We train a model from VLM-generated captions.
```

因为这些都容易被反驳。

### 13.3 推荐标题

```text
RIG-Crop: Distilling Relation-Importance Composition Reasoning for Image-Only Cropping
```

备选：

```text
Learning Image Cropping from VLM-Distilled Composition Graphs
```

---

## 14. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| Qwen bbox 不稳定 | node bbox loss 噪声大 | bbox loss 降权；只使用 high-confidence bbox；先做 utility/importance |
| caption/category 太散 | 模型学成开放类别识别 | 不训练原始文本类别，只训练 role/importance/policy |
| 和 S2CNet 太像 | reviewer 认为只是 graph cropping | 强调 explicit VLM-distilled supervision + crop-conditioned utility |
| CPC pairwise 与 Qwen utility 冲突 | loss 拉扯 | crop label 为主，utility loss 降权；只作为 auxiliary |
| 数据量不足 | graph head 过拟合 | frozen backbone + small query count；加入 GAIC/UGCrop/ProCrop |
| 推理不是 image-only | 依赖外部 detector | detector/SAM 只允许离线构建 teacher target，不允许推理调用 |

---

## 15. 最小可发表实验包

如果目标是尽快形成 AAAI 投稿雏形，最小包：

```text
Data:
  CPC Qwen train/val
  GAICD Qwen train/val

Models:
  CropRanker baseline
  RIG-Crop node-only
  RIG-Crop full

Metrics:
  CPC pairwise accuracy
  GAIC rank correlation / top-1 MOS
  efficiency
  qualitative visualization

Ablation:
  no importance
  no relation
  no utility
  heuristic vs Qwen
```

达标标准：

```text
RIG-Crop full > CropRanker baseline on CPC pairwise acc
RIG-Crop full > no-relation/no-importance/no-utility
Qwen middle state > heuristic middle state
Inference speed much faster than VLM teacher
```

---

## 16. 下一步执行清单

### 立即做

1. 检查 Qwen middle state 的 bbox/importance 覆盖率。
2. 实现 `scripts/build_middle_state_targets.py`。
3. 生成 `data/cpc_rig/metadata/{train,val}.jsonl`。
4. 先训练一个 `RIG-Crop node-only` 小模型。
5. 做 `node/importance/relation` target 可视化。

### 然后做

1. 实现 `dacc/models/rig_crop.py`。
2. 实现 `scripts/train_rig_crop.py`。
3. 跑 `CPC baseline vs RIG-Crop`。
4. 加 GAICD 混训。
5. 整理消融表和可视化。

---

## 17. 参考来源

- Good View Hunting: Learning Photo Composition From Dense View Pairs, CVPR 2018: https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Good_View_Hunting_CVPR_2018_paper.html
- A2-RL: Aesthetics Aware Reinforcement Learning for Image Cropping, CVPR 2018: https://openaccess.thecvf.com/content_cvpr_2018/html/Li_A2-RL_Aesthetics_Aware_CVPR_2018_paper.html
- Reliable and Efficient Image Cropping: A Grid Anchor Based Approach, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.html
- Image Cropping with Composition and Saliency Aware Aesthetic Score Map, AAAI 2020: https://ojs.aaai.org/index.php/AAAI/article/view/6889
- Composing Good Shots by Exploiting Mutual Relations, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Composing_Good_Shots_by_Exploiting_Mutual_Relations_CVPR_2020_paper.html
- Composing Photos Like a Photographer, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Hong_Composing_Photos_Like_a_Photographer_CVPR_2021_paper.html
- TransView: Inside, Outside, and Across the Cropping View Boundaries, ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Pan_TransView_Inside_Outside_and_Across_the_Cropping_View_Boundaries_ICCV_2021_paper.html
- Rethinking Image Cropping, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Jia_Rethinking_Image_Cropping_Exploring_Diverse_Compositions_From_Global_Views_CVPR_2022_paper.html
- Image Cropping With Spatial-Aware Feature and Rank Consistency, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Image_Cropping_With_Spatial-Aware_Feature_and_Rank_Consistency_CVPR_2023_paper.html
- Spatial-Semantic Collaborative Cropping for User Generated Content, AAAI 2024: https://ojs.aaai.org/index.php/AAAI/article/view/28303
- Learning Subject-Aware Cropping by Outpainting Professional Photos, AAAI 2024 extended arXiv: https://arxiv.org/abs/2312.12080
- Cropper: Vision-Language Model for Image Cropping through In-Context Learning, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Cropper_Vision-Language_Model_for_Image_Cropping_through_In-Context_Learning_CVPR_2025_paper.pdf
- Can Machines Understand Composition? Dataset and Benchmark for Photographic Image Composition Embedding and Understanding, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Can_Machines_Understand_Composition_Dataset_and_Benchmark_for_Photographic_Image_CVPR_2025_paper.html
- ProCrop: Learning Aesthetic Image Cropping from Professional Compositions, AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/38255
- ProCrop project page: https://bwgzk-keke.github.io/ProCrop/
