# RIGCrop 面向 AAAI 的模型方法 Transformer 化优化方案

日期：2026-06-23  
目标：以 AAAI 主会方法论文为标准，直接设计理论上更强、更像顶会方法的 RIGCrop 升级版。  
建议方法名：**RIGFormer: VLM-Distilled Relation-Importance Graph Transformer for Image-Only Cropping**  
备用短名：**RIGCrop-T++**

---

## 0. 核心判断

如果目标是投稿 AAAI，并且暂时不考虑实际实验结果约束，那么当前 `RIGCropModel` 应当升级。现在的模型更像一个可执行 proof-of-concept：

```text
Tiny CNN
  -> global pooled feature
  -> MLP unfolded node tokens
  -> pairwise MLP relations
  -> anchor crop ranker
```

这个版本能证明“VLM 中间态可以监督 image-only crop student”，但方法形态偏简单。顶会审稿人可能会认为它是：

```text
普通 crop ranker + 辅助 graph supervision + handcrafted utility features
```

因此，AAAI-facing 版本应该直接改成：

```text
Foundation visual backbone
  -> multi-scale visual tokens
  -> Transformer entity-relation graph decoder
  -> graph-aware crop decision transformer
  -> hybrid anchor-query crop proposals
  -> differentiable relation-importance utility
  -> top-k image-only crops
```

一句话：

> RIGFormer distills VLM-generated composition middle states into an image-only graph transformer that predicts entity-relation composition graphs and uses graph-conditioned crop reasoning to rank or generate aesthetic crops.

---

## 1. AAAI 论文目标重写

### 1.1 不再主打“简单可执行”

旧版 RIGCrop 的价值是跑通：

- image-only inference；
- VLM middle-state teacher；
- graph supervision；
- pairwise crop ranking。

但 AAAI 主会更需要：

- 架构上有清晰新模块；
- 模型能力与论文主张匹配；
- 能与 S2CNet、Cropper、GenCrop、ProCrop 等工作形成区分；
- 消融能够证明每个模块必要。

### 1.2 新论文主张

推荐主张：

> Existing image croppers either learn implicit visual relations or rely on costly VLM reasoning at inference. We propose RIGFormer, a VLM-distilled image-only graph transformer that learns explicit relation-importance composition graphs and conditions crop decisions on predicted relation-preservation utilities.

中文解释：

```text
已有方法要么隐式学关系，要么推理时依赖 VLM。
RIGFormer 在训练时蒸馏 VLM 显式构图推理，
推理时只输入图片，由 Transformer 预测构图图结构，
再用图结构推理每个 crop 是否保留了重要主体、关系和构图意图。
```

### 1.3 方法贡献重排

投稿时贡献建议写成四点：

1. **VLM-Distilled Composition Graph Supervision**  
   将 VLM/Qwen middle state 规范化为 entity role、importance、bbox、typed relation、action policy、crop-conditioned utility。

2. **Relation-Importance Graph Transformer**  
   用 learnable entity queries 和 relation tokens 从图像 token 中预测显式构图图结构，而不是用 pooled feature MLP 生成节点。

3. **Graph-Aware Crop Decision Transformer**  
   每个 crop token 与 predicted graph tokens 交互，显式建模“该 crop 是否保留关系、避开干扰物、避免切割主体”。

4. **Hybrid Anchor-Query Image-Only Cropping**  
   结合 dense anchors 的稳定性与 learnable crop queries 的灵活性，在推理时不依赖 VLM/detector/SAM/candidates。

---

## 2. 总体架构

### 2.1 推荐模型名

正式方法名：

```text
RIGFormer: Relation-Importance Graph Transformer
```

论文中可以写：

```text
RIGCrop-Base: current lightweight baseline.
RIGFormer: proposed AAAI-facing transformer architecture.
```

### 2.2 总体流程

```text
Input image
  |
  v
Foundation Backbone
  -> multi-scale visual tokens
  |
  v
Entity-Relation Graph Transformer Decoder
  -> entity nodes: bbox, role, importance, valid, uncertainty
  -> relation tokens: policy, weight
  -> action intent: multi-label crop policy
  |
  v
Hybrid Crop Proposal Module
  -> dense anchors
  -> learnable crop queries
  -> optional aspect-conditioned proposals
  |
  v
Graph-Aware Crop Decision Transformer
  -> crop tokens cross-attend entity/relation tokens
  -> relation-preservation attention bias
  -> crop score, utility, uncertainty
  |
  v
Diverse Top-k Crop Selection
```

训练时：

```text
image
  + CPC pairwise preferences
  + GAIC MOS-derived pairs
  + VLM/Qwen RIG targets
  + candidate utility targets
```

推理时：

```text
image only
```

---

## 3. 模块 1：Foundation Visual Backbone

### 3.1 为什么必须换强 backbone

当前 TinyBackbone 在 AAAI 论文里会显得像 smoke-test backbone。裁剪任务需要：

- semantic localization；
- object-part sensitivity；
- global composition；
- background/context understanding；
- fine-grained aesthetic cues。

强 backbone 能让方法不被质疑“提升只是因为 baseline 太弱”。

### 3.2 推荐 backbone 选择

#### 首选：DINOv2 / DINOv3-style ViT

建议：

```text
dinov2_vits14 / dinov2_vitb14
```

优点：

- object-centric 和 spatial representation 强；
- 不依赖文本输入；
- 适合 image-only student；
- 可输出 dense patch tokens；
- 与 graph decoder 天然匹配。

论文表述：

> We use a self-supervised foundation vision transformer to provide semantically aligned dense visual tokens for graph decoding.

#### 次选：CLIP ViT

优点：

- 语义强；
- 与 VLM teacher 概念空间更接近。

风险：

- CLIP 更偏 image-text alignment，空间定位未必最强；
- 审稿人可能问是否借用了语言监督。

#### 工程稳妥：Swin / ConvNeXt

优点：

- 多尺度局部视觉强；
- 训练稳定；
- 容易和 RoI/crop feature 对接。

推荐最终配置：

| 模型版本 | Backbone | 用途 |
|---|---|---|
| RIGCrop-Base | TinyBackbone | 当前 baseline |
| RIGFormer-T | Swin-T or ConvNeXt-T | 低成本主模型 |
| RIGFormer-B | DINOv2-B/14 | AAAI 主结果 |
| RIGFormer-CLIP | CLIP ViT-B/16 | 语义增强 ablation |

### 3.3 Backbone 输出

```text
F = {F_l}_{l=1..L}
```

其中：

- `F_l` 是多尺度 feature map 或 ViT patch tokens；
- 每个 token 加 2D positional encoding；
- 统一投影到 `d_model = 256 or 384`。

---

## 4. 模块 2：Entity-Relation Graph Transformer Decoder

### 4.1 为什么当前 node MLP 不够

当前模型：

```text
full_vec -> Linear -> max_nodes * graph_dim
```

问题：

- 所有节点来自一个 global vector；
- 缺少空间 token 交互；
- 节点之间没有真正的 Transformer reasoning；
- bbox 预测能力弱；
- 很难声称“graph reasoning”。

升级后：

```text
learnable entity queries
  cross-attend image tokens
  self-attend entity tokens
  iteratively refine boxes and roles
```

这更符合 AAAI 对 graph reasoning / structured prediction 的期待。

### 4.2 Entity queries

设 `M = 12 or 16` 个 learnable entity queries：

```text
Q_e ∈ R^{M x d}
```

每层 decoder：

```text
Q_e^{l+1} =
  SelfAttn(Q_e^l)
  + CrossAttn(Q_e^l, image_tokens)
  + FFN
```

输出：

```text
node_box_i        ∈ [0, 1]^4
node_role_i       ∈ {main_subject, key_object, important_background, distractor, padding}
node_importance_i ∈ [0, 1]
node_valid_i      ∈ [0, 1]
node_uncertainty_i
```

### 4.3 Iterative box refinement

每层 decoder 都预测 box delta：

```text
b_i^{l+1} = refine(b_i^l, Δb_i^l)
```

好处：

- 看起来更像 DETR 系列；
- bbox 质量更高；
- 可解释性图更可信。

### 4.4 Role-aware Hungarian matching

训练时 teacher nodes 来自 VLM middle state。因为 node 顺序不应强绑定，推荐使用 Hungarian matching：

```text
Cost(i, j) =
  λ_box * L1(box_i, box_j)
  + λ_giou * (1 - GIoU(box_i, box_j))
  + λ_role * CE(role_i, role_j)
  + λ_imp * |importance_i - importance_j|
```

匹配后计算 node losses。这样比当前固定 padding 顺序更论文级，也更鲁棒。

### 4.5 Relation tokens

当前 relation 是：

```text
pair_features(node_i, node_j) -> MLP
```

升级方案：

```text
r_ij = RelationMLP([e_i, e_j, e_i * e_j, |e_i - e_j|, geometry_ij])
```

或者更高级：

```text
relation tokens R_{ij}
  attend entity tokens
  predict relation policy and weight
```

输出：

```text
relation_policy_ij ∈ {
  none,
  preserve_relation,
  optional_preserve,
  avoid_cutting,
  leave_space,
  distractor_exclusion
}

relation_weight_ij ∈ [0, 1]
```

建议实际实现用 biaffine / MLP relation head，论文中称为 relation transformer head：

```text
p_ij = softmax(W_p [e_i, e_j, g_ij])
w_ij = sigmoid(W_w [e_i, e_j, g_ij])
```

其中 `g_ij` 是相对几何特征：

```text
dx, dy, log(w_i/w_j), log(h_i/h_j), IoU, center distance
```

---

## 5. 模块 3：Hybrid Anchor-Query Crop Proposal

### 5.1 为什么不能只用固定 anchors

固定 anchors 的问题：

- 受 grid 限制；
- 难以产生精准 top-k；
- 与 Rethinking Image Cropping / DETR-like crop methods 相比显得保守。

但完全 query-based 又有训练风险。因此建议 AAAI 版本使用 **hybrid design**：

```text
dense anchors + learnable crop queries
```

论文里可称为：

```text
Hybrid Anchor-Query Crop Proposals
```

### 5.2 Anchor branch

保留 GAIC-style anchors：

```text
scales = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5}
aspect_ratios = {1:1, 4:3, 3:4, 16:9, 9:16, original}
grid = 7 or 9
```

产生 `N_a = 300-800` 个 candidates。

每个 anchor crop token：

```text
c_a = MLP([RoIAlign(F, box), box_embedding, aspect_embedding])
```

### 5.3 Query branch

加入 `K = 16 or 32` 个 learnable crop queries：

```text
Q_c ∈ R^{K x d}
```

通过 crop decoder 预测：

```text
crop_box_k ∈ [0, 1]^4
crop_score_k
crop_quality_k
```

训练方式：

- 对 GAIC：匹配 MOS 高的 candidate；
- 对 CPC：用 pairwise preference 形成 winner/loser supervision；
- 对所有数据：加 diversity loss，避免 top-k 塌缩。

### 5.4 为什么 hybrid 最适合 AAAI

| 方案 | 优点 | 缺点 |
|---|---|---|
| anchors only | 稳定、易对比 GAIC | 不够新 |
| queries only | 新、灵活 | 训练风险高 |
| hybrid | 稳定 + 高级 + 可消融 | 实现稍复杂 |

AAAI 论文可以写：

> The anchor branch provides dense coverage and stable preference learning, while the query branch learns adaptive crop hypotheses beyond predefined grids.

---

## 6. 模块 4：Graph-Aware Crop Decision Transformer

### 6.1 当前 score head 的不足

当前：

```text
score = MLP([full_vec, crop_vec, box_vec, graph_feat])
```

这能跑，但不够像“reasoning”。升级后每个 crop token 应与 entity/relation graph 做 attention。

### 6.2 Crop token construction

对每个 candidate crop `c`：

```text
t_c = MLP([
  crop_visual_feature,
  box_positional_embedding,
  aspect_embedding,
  scale_embedding
])
```

crop visual feature 可来自：

- crop patch backbone；
- RoIAlign from image feature map；
- token pooling inside crop region。

推荐优先使用 RoIAlign/token pooling，避免每个 crop 单独跑 backbone，效率更高。

### 6.3 Crop-to-graph cross-attention

核心模块：

```text
T_c' = CrossAttn(T_c, E_nodes)
T_c'' = CrossAttn(T_c', R_relations)
T_c_out = FFN(T_c'')
```

含义：

- crop token 查询哪些 entity 被保留；
- crop token 查询哪些 relation 被破坏；
- relation token 告诉 crop token 该保留、可丢弃或要避开什么。

### 6.4 Relation-preservation attention bias

为了不只是普通 attention，加入 crop-conditioned bias：

```text
bias(c, i) = coverage(box_i, crop_c) * importance_i
```

relation bias：

```text
bias(c, i, j) =
  coverage(box_i, crop_c)
  * coverage(box_j, crop_c)
  * relation_weight_ij
  * preserve_prob_ij
```

这些 bias 可以作为 attention logit bias：

```text
Attn(q_c, k_i) = softmax(q_c k_i / sqrt(d) + β * bias(c, i))
```

论文亮点：

> Crop decisions are not only conditioned on graph embeddings but explicitly biased by differentiable relation-preservation estimates.

### 6.5 输出 heads

每个 crop token 输出：

```text
score_c       ∈ [0, 1]
utility_c     ∈ [0, 1]
uncertainty_c ∈ [0, 1]
action_c      multi-label optional
```

最终 score：

```text
final_score_c =
  score_head(t_c_out)
  + α * utility_head(t_c_out)
  - γ * uncertainty_c
```

也可以只把 utility 作为 auxiliary，避免公式太硬。

---

## 7. 模块 5：Differentiable Relation-Importance Utility

### 7.1 保留旧版最有辨识度的部分

当前 RIGCrop 最有原创感的是：

```text
node_keep
relation_keep
distractor
boundary
main_cov
```

升级版不要删除，而是改成：

```text
Differentiable Relation-Importance Utility Layer
```

简称：

```text
DRIU / DRPU
```

推荐名称：

```text
Differentiable Relation Preservation Utility (DRPU)
```

### 7.2 Utility components

对 crop `c` 和 predicted graph `G`：

```text
U_node(c) =
  Σ_i valid_i * importance_i * non_distractor_i * coverage(b_i, c)

U_rel(c) =
  Σ_{i,j} preserve_prob_ij * relation_weight_ij
  * coverage(b_i, c) * coverage(b_j, c)

U_cut(c) =
  Σ_i importance_i * non_distractor_i
  * boundary_cut(b_i, c)

U_dist(c) =
  Σ_i importance_i * distractor_prob_i * coverage(b_i, c)

U_main(c) =
  Σ_i main_prob_i * subject_position_score(b_i, c)
```

最终：

```text
U(c, G) =
  MLP([U_node, U_rel, U_cut, U_dist, U_main, box_feat])
```

这比旧版固定线性公式更高级，同时保留可解释性。

### 7.3 Utility supervision

teacher utility 来自 `rig_targets.candidate_utilities`。训练：

```text
L_utility =
  SmoothL1(U_pred, U_teacher)
  + PairwiseRank(U_pred_winner, U_pred_loser)
```

可再加 consistency：

```text
L_score_utility_consistency =
  max(0, margin - sign(y_i - y_j) * (U_i - U_j))
```

---

## 8. Loss 设计

总 loss：

```text
L =
  λ_rank L_rank
  + λ_list L_list
  + λ_node L_node
  + λ_rel L_rel
  + λ_util L_util
  + λ_action L_action
  + λ_query L_query_set
  + λ_div L_diversity
  + λ_cons L_consistency
  + λ_unc L_uncertainty
```

### 8.1 Crop ranking loss

保留 pairwise Bradley-Terry / softplus ranking：

```text
L_rank =
  softplus(-(s_w - s_l)) * weight
```

适合 CPC 和 GAIC-derived pairs。

### 8.2 Listwise ranking loss

对 GAICD 有 MOS 的候选列表，加入 listwise：

```text
p_i = softmax(s_i / τ)
q_i = softmax(MOS_i / τ)
L_list = KL(q || p)
```

或者 SoftNDCG / ListNet。AAAI 方法论文中 listwise ranking 会比只用 pairwise 更完整。

### 8.3 Node loss

Hungarian matching 后：

```text
L_node =
  L1(box)
  + GIoU(box)
  + CE(role)
  + SmoothL1(importance)
  + BCE(valid)
```

### 8.4 Relation loss

匹配 node 后计算：

```text
L_rel =
  CE(relation_policy)
  + SmoothL1(relation_weight)
```

只在 valid relation mask 上计算。

### 8.5 Action loss

```text
L_action = BCEWithLogits(action_logits, action_targets)
```

action 不一定作为主贡献，但可以增强 teacher distillation。

### 8.6 Query crop set loss

对 query branch：

```text
L_query_set =
  HungarianMatch(pred_crop_queries, positive_crops)
  + L1/GIoU box loss
  + ranking/objectness loss
```

如果没有 ground-truth single crop，则使用：

- top MOS candidates；
- high preference winners；
- best-of-k positive set。

### 8.7 Diversity loss

避免 top-k crop 全部相似：

```text
L_div =
  Σ_{i≠j} max(0, IoU(c_i, c_j) - δ)
```

或使用 Determinantal Point Process style diversity。

### 8.8 Uncertainty-aware teacher distillation

VLM teacher 有噪声。建议加入 teacher confidence：

```text
L_teacher = confidence * L_supervision
```

如果没有显式 confidence，可从 schema audit 派生：

- bbox 是否存在；
- relation 是否来自 explicit preserve/avoid text；
- VLM 是否输出 unknown；
- importance 是否为默认值。

---

## 9. 数据 Schema 升级

### 9.1 max_nodes 升级

当前默认 `max_nodes=8`，建议 AAAI 版本改为：

```text
max_nodes = 12 or 16
```

理由：

- 复杂 UGC 图像需要更多实体；
- S2CNet/UGC 场景强调多主体复杂背景；
- 可以覆盖 main/key/background/distractor。

### 9.2 新增 teacher quality 字段

建议在 `rig_targets` 中加入：

```json
{
  "teacher_quality": {
    "has_main_subject_bbox": true,
    "node_bbox_rate": 0.75,
    "relation_source": "explicit_intent|default_rule",
    "action_unknown_rate": 0.0,
    "teacher_confidence": 0.83
  }
}
```

训练中作为 loss weight。

### 9.3 relation triplets

除了 matrix，可额外保存 triplets：

```json
{
  "relation_triplets": [
    {
      "subject": 0,
      "object": 1,
      "policy": "preserve_relation",
      "weight": 0.9,
      "source": "vlm_preserve_text"
    }
  ]
}
```

好处：

- 方便可视化；
- 方便论文解释；
- 方便 relation transformer 训练。

### 9.4 no-leakage audit

为了防守审稿人质疑，必须保留：

```text
VLM prompt input:
  image + caption + semantic_type only

Not used by VLM:
  candidate boxes
  MOS
  best_crop
  pairwise preferences
```

建议新增审计脚本：

```text
RIGCrop/scripts/audit_no_label_leakage.py
```

输出：

```json
{
  "vlm_prompt_fields": ["image", "caption", "semantic_type"],
  "forbidden_fields_in_prompt": [],
  "passed": true
}
```

---

## 10. 训练策略

### 10.1 四阶段训练

#### Stage A：Graph warm-up

冻结 backbone 或低学习率，只训练 graph decoder：

```text
L = L_node + L_rel + L_action
```

目的：

- 先让 model 学会预测 composition graph；
- 避免 ranking loss 过早主导。

#### Stage B：Crop ranking warm-up

训练 anchor branch + graph-aware scorer：

```text
L = L_rank + L_util + L_node + L_rel
```

目的：

- 让 crop scorer 使用 predicted graph；
- 保持 teacher graph 监督。

#### Stage C：Hybrid query joint training

加入 query crop branch：

```text
L = full objective
```

目的：

- 学 adaptive crop proposal；
- 提升 top-k 多样性和精度。

#### Stage D：Hard negative mining + fine-tuning

挖掘：

- 高 score 但 cut subject 的 crop；
- 保留 distractor 的 crop；
- 丢失 key object relation 的 crop；
- full image over-conservative crop。

加入 hard pairs：

```text
winner = human/VLM-preferred crop
loser = model high-score failure crop
```

### 10.2 推荐超参数

| 参数 | 建议 |
|---|---|
| image size | 384 / 448 / 518 |
| d_model | 256 for Swin-T, 384 for DINOv2-B |
| entity queries | 12 or 16 |
| crop queries | 16 or 32 |
| graph decoder layers | 4-6 |
| crop decision transformer layers | 2-4 |
| anchors per image | 300-800 |
| optimizer | AdamW |
| lr backbone | 1e-5 |
| lr new modules | 1e-4 |
| weight decay | 0.05 for ViT-style, 1e-4 for small CNN |
| warm-up epochs | 3-5 |
| total epochs | 30-80 |

---

## 11. 推理流程

推理输入：

```text
image only
```

推理禁止：

```text
Qwen / VLM / caption / middle-state JSON / detector / SAM / dataset candidates
```

流程：

```text
image
  -> foundation backbone
  -> entity-relation graph decoder
  -> anchors + crop queries
  -> graph-aware crop scorer
  -> merge anchor/query candidates
  -> diversity-aware NMS
  -> top-k crops
```

最终输出：

```json
{
  "topk": [
    {
      "box": [x1, y1, x2, y2],
      "score": 0.91,
      "utility": 0.87,
      "reason": {
        "main_subject_keep": 0.94,
        "relation_keep": 0.82,
        "distractor_penalty": 0.05,
        "boundary_penalty": 0.02
      }
    }
  ]
}
```

`reason` 可选，但强烈建议用于论文可视化。

---

## 12. 与已有工作的差异写法

### 12.1 对 S2CNet

不要写：

```text
We model spatial-semantic relations.
```

应写：

```text
Unlike prior relation-aware croppers that learn implicit spatial-semantic graphs from crop labels, RIGFormer distills explicit VLM-generated composition middle states, including typed preservation policies and relation-aware crop utility, into an image-only transformer.
```

### 12.2 对 Cropper

不要写：

```text
We use VLM for cropping.
```

应写：

```text
Cropper performs VLM reasoning at inference. RIGFormer uses VLM only offline as a teacher and deploys a compact image-only student, enabling efficient large-scale cropping without online multimodal inference.
```

### 12.3 对 GenCrop / ProCrop

不要写：

```text
We use weak supervision.
```

应写：

```text
Rather than generating only weak crop labels from professional compositions, RIGFormer supervises the intermediate reasoning structure behind crop decisions: what entities matter, which relations should be preserved, and how crop boundaries affect those relations.
```

### 12.4 对 GAIC / CPC

不要写：

```text
We train on CPC and GAIC.
```

应写：

```text
We unify heterogeneous crop supervision through pairwise ranking while introducing VLM-distilled graph supervision as an orthogonal reasoning signal.
```

---

## 13. 消融实验设计

### 13.1 模块消融

| Variant | 目的 |
|---|---|
| CropRanker baseline | 普通 image+crop+box ranker |
| RIGCrop-Base | 当前 Tiny CNN + graph utility |
| RIGFormer w/o VLM graph | 验证 VLM graph supervision |
| RIGFormer w/o relation decoder | 验证 relation modeling |
| RIGFormer w/o DRPU | 验证 explicit utility |
| RIGFormer w/o crop decision transformer | 验证 crop-graph attention |
| RIGFormer anchors only | 验证 query branch |
| RIGFormer queries only | 验证 hybrid design |
| RIGFormer w/o uncertainty weighting | 验证 teacher noise handling |
| RIGFormer full | 主模型 |

### 13.2 Backbone 消融

| Backbone | 目的 |
|---|---|
| Tiny CNN | 当前 baseline |
| ConvNeXt-T | 强 CNN |
| Swin-T | hierarchical transformer |
| DINOv2-S/B | self-supervised foundation token |
| CLIP ViT-B | semantic alignment |

### 13.3 Teacher 消融

| Teacher | 目的 |
|---|---|
| no teacher | 纯 crop supervision |
| heuristic teacher | 低质量中间态 |
| Qwen teacher | 主实验 |
| OpenAI/alternative VLM teacher | teacher 泛化 |
| noisy teacher | 鲁棒性 |

### 13.4 Utility 消融

| Utility | 目的 |
|---|---|
| handcrafted 5-dim utility | 当前版本 |
| learned DRPU | 主模型 |
| no relation_keep | 验证关系保留 |
| no distractor penalty | 验证干扰物 |
| no boundary penalty | 验证避免切割 |
| utility auxiliary only | 验证是否应直接进 score |
| utility score fusion | 验证融合策略 |

---

## 14. 论文图表设计

### 14.1 Figure 1：总体架构

必须画出：

```text
Training:
  VLM middle state -> RIG targets -> graph supervision

Inference:
  image only -> RIGFormer -> top-k crops
```

重点用颜色区分：

- teacher-only path；
- image-only inference path；
- graph decoder；
- crop decision transformer；
- DRPU utility。

### 14.2 Figure 2：Graph-aware crop reasoning

展示一张图：

- predicted main subject；
- key objects；
- distractor；
- relation lines；
- 3 个 crop candidates；
- 每个 crop 的 node_keep / relation_keep / boundary / distractor 分数。

### 14.3 Figure 3：与 Cropper 的部署对比

```text
Cropper:
  image + prompt -> VLM inference -> iterative crop

RIGFormer:
  image -> student -> crop
```

展示 latency / cost / parameters / online dependency。

### 14.4 Tables

| 表 | 内容 |
|---|---|
| Table 1 | CPC/GAIC main comparison |
| Table 2 | Cross-dataset generalization |
| Table 3 | Ablation study |
| Table 4 | Backbone and architecture variants |
| Table 5 | Efficiency comparison |
| Table 6 | Human preference study |

---

## 15. 推荐代码结构

建议不要直接覆盖当前 `RIGCropModel`。保留它作为 baseline，新建 transformer 版本：

```text
RIGCrop/rigcrop/
  model.py                     # keep RIGCrop-Base
  rigformer.py                 # new main model
  backbones.py                 # DINOv2/Swin/ConvNeXt/CLIP wrapper
  transformer_blocks.py        # decoder layers, attention bias
  graph_decoder.py             # entity-relation graph transformer
  crop_proposal.py             # hybrid anchors + crop queries
  crop_decision_transformer.py # crop-graph cross-attention scorer
  utility.py                   # differentiable relation-preservation utility
  matching.py                  # Hungarian matching for nodes/crops
  losses_rigformer.py          # full objective

RIGCrop/scripts/
  train_rigformer.py
  eval_rigformer.py
  predict_rigformer.py
  visualize_rig_graph.py
  audit_no_label_leakage.py

RIGCrop/configs/
  rigformer_dinov2_cpc.yaml
  rigformer_dinov2_cpc_gaic.yaml
  rigformer_swin_cpc_gaic.yaml
  ablations/
    rigformer_no_relation.yaml
    rigformer_no_drpu.yaml
    rigformer_anchors_only.yaml
    rigformer_queries_only.yaml
    rigformer_no_vlm_graph.yaml
```

---

## 16. 推荐实现优先级

### Phase 1：最小 AAAI 架构升级

必须做：

1. 强 backbone wrapper；
2. Transformer entity graph decoder；
3. graph-aware crop scorer；
4. learned DRPU；
5. 保留 anchors。

这已经足够把方法从 RIGCrop-Base 升级到 RIGFormer。

### Phase 2：更强顶会形态

继续做：

1. query crop branch；
2. hybrid anchor-query fusion；
3. diversity-aware top-k；
4. uncertainty-aware teacher weighting。

### Phase 3：论文增强

再做：

1. graph visualization；
2. no-leakage audit；
3. teacher quality audit；
4. human preference interface。

---

## 17. 最终推荐方法版本

如果只能选择一个主模型，建议：

```text
RIGFormer-B

Backbone:
  DINOv2-B/14 or Swin-B

Graph:
  16 entity queries
  6 graph decoder layers
  relation biaffine head

Crop:
  500 dense anchors
  16 learned crop queries
  graph-aware crop decision transformer, 3 layers

Utility:
  learned DRPU with explicit components

Loss:
  pairwise + listwise + node + relation + utility + action + diversity

Inference:
  image-only
```

如果算力紧张，降级：

```text
RIGFormer-T

Backbone:
  Swin-T / ConvNeXt-T / DINOv2-S

Graph:
  12 entity queries
  4 graph decoder layers

Crop:
  anchors only first
  query branch optional
```

---

## 18. 论文摘要级方法描述

可以直接作为论文方法段落雏形：

> We propose RIGFormer, a VLM-distilled graph transformer for image-only cropping. During training, a multimodal teacher parses each image into a structured composition middle state, including entity roles, importance scores, bounding boxes, typed relation-preservation policies, and crop-conditioned utility targets. RIGFormer learns to predict this composition graph from image tokens alone using a transformer entity-relation decoder. Candidate crops are represented as crop tokens and refined by a graph-aware crop decision transformer, where attention is biased by differentiable relation-preservation estimates. The model is trained with heterogeneous crop supervision from comparative preferences and MOS-derived rankings, together with graph, relation, action, and utility distillation losses. At inference, RIGFormer requires only a single image and produces diverse top-k crops without invoking the VLM teacher, detectors, segmentation models, or dataset candidates.

中文摘要：

> 我们提出 RIGFormer，一个面向 image-only 裁剪的 VLM 蒸馏图 Transformer。训练时，多模态 teacher 将图像解析为结构化构图中间态，包括实体角色、重要性、边界框、关系保留策略和 crop-conditioned utility。RIGFormer 从图像 token 中预测该构图图结构，并让 crop token 通过 graph-aware decision transformer 查询实体和关系，从而判断候选裁剪是否保留主体、关键关系并排除干扰物。推理时模型只输入图像，不调用 VLM、detector、SAM 或中间态 JSON。

---

## 19. 对当前代码的直接映射

| 当前实现 | AAAI 升级 |
|---|---|
| `TinyBackbone` | DINOv2/Swin/ConvNeXt backbone |
| `full_vec -> graph_proj -> node_tokens` | entity queries + transformer graph decoder |
| `relation_head(pair_features)` | relation biaffine / relation transformer head |
| `generate_anchors` | hybrid anchors + learned crop queries |
| crop patch backbone | RoI/token pooling + crop token embedding |
| `score_head(concat(...))` | graph-aware crop decision transformer |
| 5-dim graph feature | learned DRPU with explicit utility components |
| fixed node order loss | Hungarian node matching |
| pairwise only | pairwise + listwise + set prediction + diversity |
| pairwise acc only | MOS/NDCG/IoU/top-k/human preference/efficiency |

---

## 20. 最终建议

为了 AAAI，不建议只在当前模型上“小修小补”。最合理策略是：

```text
保留当前 RIGCrop 作为 Base 和 ablation。
新增 RIGFormer 作为主方法。
把论文创新从“我们用了 VLM 中间态”升级为：

VLM-distilled explicit composition graph
  + transformer entity-relation decoding
  + graph-aware crop decision
  + differentiable relation-preservation utility
  + image-only deployment
```

这套方法从理论形态上更符合 AAAI：

- 有结构化推理；
- 有多模态 teacher-student distillation；
- 有 Transformer 架构创新；
- 有可解释 utility；
- 有高效 image-only inference；
- 有清楚消融路径；
- 能与 S2CNet、Cropper、GenCrop、ProCrop 形成差异。

一句话总结：

> 当前 RIGCrop 是一个好想法的可执行 baseline；RIGFormer 才是更适合 AAAI 主会叙事的模型方法。
