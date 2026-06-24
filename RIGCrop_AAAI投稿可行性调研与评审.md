# RIGCrop AAAI 投稿可行性调研与模拟评审报告

日期：2026-06-23  
工作区：`/Users/hmx/XMU/image_cut`  
使用框架：`academic-research-suite` 深度调研 + `academic-paper-reviewer` 多视角模拟评审  
重点对象：`RIGCrop` 方法、DACC/CPC/GAIC 数据 pipeline、VLM/Qwen 中间态蒸馏、AAAI 主会投稿可行性

---

## 0. 结论先行

### 0.1 当前判断

**以当前代码和证据状态，RIGCrop 还不够稳妥地直接投稿 AAAI 主会。** 主要原因不是想法弱，而是论文证据链尚未完成：仓库中目前没有真实 CPC/GAIC/RIG 数据文件、没有完整训练 history、没有与 S2CNet、TransView、GAIC、Cropper、GenCrop、ProCrop 等强基线的实证对比。当前可确认的是：方法原型、数据契约、target 构建、训练脚本、消融配置和 smoke test 是可执行的；但 AAAI 主会需要的是可复现、可比较、能打动审稿人的完整实验论文。

**如果 2026-07-28 AAAI-27 全文截止前能够补齐真实数据实验、强基线、消融、人评和失败案例分析，RIGCrop 有机会以高风险投稿冲击 AAAI。** 但以 2026-06-23 计，距离 AAAI-27 摘要截止 2026-07-21、全文截止 2026-07-28、补充材料和代码截止 2026-07-31 只剩约 4-5 周；如果没有现成数据和 GPU 训练结果，主会成功率偏低。

### 0.2 推荐定位

最强论文主张应收敛为：

> **RIGCrop learns an explicit relation-importance composition graph from VLM-generated middle states during training, then performs image-only crop ranking at inference by conditioning candidate scores on predicted graph-preservation utilities.**

不要把创新写成“使用 VLM 做裁剪”或“考虑多主体关系”。这两个点已经被近期工作严重占据。RIGCrop 的可投稿核心是：

1. **显式 VLM 构图中间态蒸馏**：role、importance、bbox、typed relation、crop policy、action target。
2. **推理阶段 image-only**：不调用 Qwen/VLM，不读中间态 JSON，不依赖 detector/SAM。
3. **crop-conditioned relation utility**：把节点覆盖、关系保留、干扰物覆盖、边界切割、主体覆盖显式接入 score head。
4. **跨 CPC pairwise 与 GAIC MOS-derived pair 的统一训练管线**：保留 pairwise ranking 形式，避免把 CPC pseudo score 与 GAIC MOS 粗暴混成同一回归标签。

### 0.3 投稿建议

| 方案 | 建议 | 理由 |
|---|---|---|
| AAAI-27 主会立即投稿 | **高风险，可冲但不稳** | idea 有潜力，但当前缺真实结果、强基线、人评、泛化实验 |
| AAAI workshop / CVPR workshop / arXiv 技术报告 | **稳妥** | 当前原型和设计足以形成方法型报告 |
| 补齐实验后投 AAAI-27 | **有条件推荐** | 需要在 4-5 周内完成完整实验矩阵 |
| 延后一轮投 CVPR/ICCV/AAAI | **最优学术路线** | 可升级 backbone/query generator、补大规模数据和解释性实验 |

---

## 1. Evidence Passport

### 1.1 本地代码证据

已检查的核心文件：

| 模块 | 文件 | 证据点 |
|---|---|---|
| RIGCrop 模型 | `RIGCrop/rigcrop/model.py` | graph prediction、relation matrix、graph_features_for_crop、score head |
| RIG target 构建 | `RIGCrop/rigcrop/schema.py` | VLM middle state -> nodes/relations/utilities/actions |
| 数据集 | `RIGCrop/rigcrop/data.py` | CPC pairwise + GAIC MOS-derived pairs + RIG tensors |
| 损失 | `RIGCrop/rigcrop/losses.py` | crop pair、node、relation、utility、action losses |
| 训练 | `RIGCrop/scripts/train_rig_crop.py` | ConcatDataset、DDP、checkpoint、loss aggregation |
| 评估 | `RIGCrop/scripts/eval_rig_crop.py` | pairwise accuracy / weighted accuracy |
| 推理 | `RIGCrop/scripts/predict_rig_crop.py` | image-only anchors -> top-k crops |
| VLM enrichment | `scripts/enrich_dacc_with_vlm_semantics.py`、`scripts/enrich_gaic_with_vlm_semantics.py`、`composition_dataset_builder/vlm.py` | Qwen/OpenAI prompt 与 middle state 字段 |
| 上游数据转换 | `scripts/cpc_to_dacc_jsonl.py`、`scripts/gaic_to_dacc_jsonl.py` | CPC/GAIC 转 DACC-style JSONL |
| 执行文档 | `RIGCrop/docs/TECHNICAL_DESIGN.md`、`RIGCrop/docs/DATA_PIPELINE_CONTRACT.md`、`RIGCrop/docs/EXECUTION.md` | 方法目标、管线契约、训练命令 |

### 1.2 当前本地证据状态

运行检查发现：

- 仓库中未发现真实 `data/cpc_rig`、`data/gaic_rig`、`data/cpc_semantic_qwen` 或 `data/gaic_semantic_qwen` JSONL。
- 未发现真实训练 checkpoint/history。
- 已运行 `bash RIGCrop/scripts/run_smoke_test.sh`，结果通过：
  - target 构建成功；
  - 1 epoch smoke 训练成功；
  - smoke validation pairwise accuracy = 0.8333；
  - image-only predict 成功输出 top-3 crop。

注意：smoke test 只能证明代码链路可执行，不能证明方法优于已有研究。

### 1.3 外部文献证据

优先使用官方会议页、CVF Open Access、AAAI OJS、项目页或论文页。检索日期为 2026-06-23。主要参考源列在第 9 节。

---

## 2. 现有研究格局

### 2.1 自动图像裁剪的主线

自动图像裁剪大致经历了五条路线：

1. **美学/显著性驱动 scoring**：通过 saliency、aesthetic score map 或 composition map 估计裁剪质量。
2. **候选框 ranking**：先产生 crop candidates，再回归/排序候选框质量。
3. **pairwise preference learning**：使用 view pair 偏好，训练“哪个裁剪更好”。
4. **关系/构图建模**：把 crop 内外、边界、主体和背景的关系纳入判断。
5. **VLM/MLLM 裁剪与弱监督**：用 VLM 推理、扩图/外绘或专业照片检索构造弱标签。

RIGCrop 同时踩在第 3、4、5 条线上，因此容易有创新空间，也容易被审稿人要求与这些线上的强工作逐一对比。

### 2.2 文献矩阵

| 工作 | Venue | 关键机制 | 对 RIGCrop 的压力 | RIGCrop 可区分点 |
|---|---|---|---|---|
| A2-RL | CVPR 2018 | 将裁剪建模为序列决策，用 aesthetics-aware reward 学习动作 | 说明裁剪可以是 policy/action 问题 | RIGCrop 不是在线 RL，而是图监督蒸馏 + ranking |
| Good View Hunting / CPC | CVPR 2018 | 大规模 comparative view pairs，用 pairwise 方式学构图偏好 | CPC 已经证明 pairwise 学习有效 | RIGCrop 继承 pairwise，但增加结构化构图解释 |
| GAIC / GAICD | CVPR 2019 | grid anchor candidate scoring，候选框有 MOS | anchor scoring 已很成熟 | RIGCrop 的新意在 graph-conditioned scoring，不是 anchor 本身 |
| Composition + Saliency Aesthetic Map | AAAI 2020 | saliency 与 composition-aware aesthetic map | 显式构图特征已有先例 | RIGCrop 是 instance-level role/relation graph，不是 dense map |
| Composing Photos Like a Photographer | CVPR 2021 | key composition map，学习专业摄影构图 | 构图知识显式编码已有强先例 | RIGCrop 聚焦实体关系、裁剪边界和 VLM teacher |
| TransView | ICCV 2021 | crop 内、外、跨边界 visual words 的 attraction/repulsion | crop-conditioned relation 已有先例 | RIGCrop 的 relation 是 typed VLM-distilled semantic relation |
| Rethinking Image Cropping | CVPR 2022 | 从全局视角探索多样构图，learnable anchors / diverse crops | top-k 多样输出已有强工作 | 当前 RIGCrop 还是固定 anchors，需补多样性或承认边界 |
| Spatial-aware Feature + Rank Consistency | CVPR 2023 | crop 与 aesthetic elements 的空间关系 + rank consistency | 空间关系和 rank consistency 已成熟 | RIGCrop 可把 aesthetic elements 升级为 VLM entity graph |
| S2CNet | AAAI 2024 | spatial-semantic collaborative network，复杂 UGC 场景多主体关系 | 最接近“多主体 + 关系”方向 | RIGCrop 必须强调显式 VLM 中间态监督，而非仅关系建模 |
| GenCrop | AAAI 2024 | professional photos + diffusion outpainting 构造 subject-aware 弱监督 | 弱监督数据构造已有 AAAI 先例 | RIGCrop 学 relation/importance/policy，不只是 subject-aware crop |
| Cropper | CVPR 2025 | VLM in-context learning + iterative refinement 做裁剪 | “VLM 能裁剪”已被占据 | RIGCrop 推理不依赖 VLM，可部署为轻量 image-only student |
| PICD | CVPR 2025 | 专业摄影构图类别数据集与 benchmark | composition understanding benchmark 趋势增强 | RIGCrop 可用作构图理解辅助评估，但不是直接裁剪标签 |
| ProCrop | AAAI 2026 | 专业构图检索与大规模弱标注 crop 数据 | 专业构图弱监督很新，AAAI 认可此方向 | RIGCrop 的差异是 VLM middle-state graph distillation |

### 2.3 与 RIGCrop 最相关的竞争面

#### 2.3.1 S2CNet 是最大“关系建模”竞争者

S2CNet 已经在 AAAI 2024 把 UGC image cropping 推向 spatial-semantic collaborative graph/network 方向。如果 RIGCrop 只声称“考虑多个主体关系”，审稿人很容易认为是已有工作延伸。

RIGCrop 应这样写差异：

> Existing relation-aware croppers mostly learn implicit visual/candidate relations. RIGCrop distills explicit VLM composition reasoning into supervised latent graph targets, including entity role, importance, typed crop policy, and crop-conditioned preservation utility.

#### 2.3.2 Cropper 是最大“VLM 裁剪”竞争者

Cropper 已经说明 VLM 可以通过 in-context learning 和 iterative refinement 做 image cropping。RIGCrop 不能再把“VLM 会分析构图”当创新。必须强调：

- VLM 只在离线训练数据构建阶段作为 teacher；
- 推理阶段不调用 VLM；
- student 学到的是可微的 latent composition graph；
- 计算成本、延迟、部署场景与 VLM-inference 方法不同。

#### 2.3.3 GenCrop/ProCrop 是最大“弱监督数据构造”竞争者

GenCrop 和 ProCrop 说明用 professional composition/outpainting/retrieval 构造大规模弱监督 crop 数据已经是 AAAI 可接受方向。RIGCrop 若只写“我们用 VLM 生成伪标签”，会显得不够新。

应写成：

> The teacher does not directly provide final crop labels. It provides a decomposed composition state that supervises why a crop should preserve or remove certain entities and relations.

---

## 3. 当前 RIGCrop 方法架构分析

### 3.1 方法一句话

RIGCrop 当前实现是一个 **image-only candidate crop ranker with latent relation-importance graph prediction**：

```text
image
  -> visual backbone
  -> latent node tokens
      -> node boxes / roles / importance / valid
      -> pairwise relation policy / relation weight
      -> action targets
  -> candidate crop branch
      -> full-image feature + crop feature + box feature + graph_features_for_crop
  -> crop score + utility
```

### 3.2 模型结构

`RIGCropModel` 的核心路径在 `RIGCrop/rigcrop/model.py`：

- `TinyBackbone` 提取整图和 crop patch 特征。
- `graph_proj` 把整图 pooled feature 映射为 `max_nodes` 个 latent node token。
- node heads 输出：
  - `node_box`
  - `node_role`
  - `node_importance`
  - `node_valid`
- relation heads 对 node pair 特征输出：
  - `relation_logits`
  - `relation_weight`
- action head 输出 multi-label action logits。
- candidate branch 输入：
  - full image vector
  - crop patch vector
  - 8 维 box feature
  - 5 维 graph feature
- score head 输出候选 crop score。

关键证据：

- `encode_graph` 从图像预测 latent graph：`RIGCrop/rigcrop/model.py:94`。
- `forward` 中 `graph_features_for_crop` 被拼入 score head：`RIGCrop/rigcrop/model.py:116`。
- `graph_features_for_crop` 显式计算 node_keep、relation_keep、distractor、boundary、main_cov：`RIGCrop/rigcrop/model.py:128`。

这说明当前代码确实不是“只训练时多一个 caption”，而是让预测图参与打分。

### 3.3 graph_features_for_crop 的意义

当前 5 维 graph feature：

| 特征 | 代码含义 | 构图语义 |
|---|---|---|
| `node_keep` | 非 distractor 节点 coverage * importance | 保留重要主体/物体 |
| `relation_keep` | pair coverage * preserve probability * relation weight | 保留主体-物体/背景关系 |
| `distractor` | distractor coverage * importance | 惩罚保留干扰物 |
| `boundary` | coverage*(1-coverage)*importance | 惩罚裁切主体/重要物体边界 |
| `main_cov` | main_subject 概率加权 coverage | 主体覆盖程度 |

这是 RIGCrop 最有论文价值的代码点：crop score 不是只看视觉 patch，而是看 predicted graph 与 candidate crop 的相容性。

### 3.4 当前架构的弱点

| 弱点 | 对 AAAI 的影响 | 建议 |
|---|---|---|
| Backbone 是 Tiny CNN | 容易被认为实验上限低，不能与 SOTA 公平竞争 | 增加 ConvNeXt-T/Swin-T/DINOv2-S 或 CLIP ViT backbone |
| node tokens 来自全局 pooled feature，不是 Transformer decoder | graph 表达能力有限 | 加 cross-attention node queries 或 DETR-style object queries |
| 推理候选框是固定 grid anchors | 多样性和精细边界弱于 learnable crop queries | 增加 query-based crop generator 或至少更密 anchor + NMS |
| 评估只实现 pairwise accuracy | AAAI 证据不够 | 补 Top-1 IoU、Top-k recall、NDCG、Spearman/Kendall、人评 |
| relation policy 来自规则归一化文本 | 容易被质疑 teacher 噪声和规则脆弱 | 加 schema audit、teacher agreement、人审子集 |
| utility 是手工公式 | 可能被认为启发式拼接 | 做 utility/no-utility、relation/no-relation、learned utility 对照 |

---

## 4. 数据 Pipeline 分析

### 4.1 总体 pipeline

当前 pipeline 可概括为：

```text
CPC raw / GAICD raw
  -> DACC-style JSONL
  -> VLM/Qwen semantic enrichment
  -> RIG target construction
  -> RIGPairwiseDataset
  -> RIGCrop training
  -> image-only anchor prediction
```

### 4.2 CPC pipeline

代码路径：

- `scripts/cpc_to_dacc_jsonl.py`
- `scripts/enrich_dacc_with_vlm_semantics.py`
- `RIGCrop/scripts/build_middle_state_targets.py`

CPC 数据被转换为 DACC-style JSONL，核心字段包括：

- `sample_id`
- `image_path`
- `image_width`
- `image_height`
- `candidates`
- `pairwise_preferences`
- `cpc_supervision`

RIGCrop 数据集直接读取 CPC 的 `pairwise_preferences`，符合 CPC 原始 comparative view pairs 的性质。

### 4.3 GAICD pipeline

代码路径：

- `scripts/gaic_to_dacc_jsonl.py`
- `scripts/enrich_gaic_with_vlm_semantics.py`
- `RIGCrop/scripts/build_middle_state_targets.py`

GAICD 原始候选框包含 MOS。当前代码不把 MOS 与 CPC pseudo score 直接混成一个统一回归标签，而是在没有 pairwise preference 时从 candidate scores 派生 pair：

- `RIGCrop/rigcrop/data.py:48` 先读显式 `pairwise_preferences`。
- `RIGCrop/rigcrop/data.py:50` 在没有 pairwise 时调用 `_derive_pairs_from_candidate_scores`。
- `RIGCrop/rigcrop/data.py:88` 起把 MOS/final_score 转成 pairwise preference。

这是一个正确的 pipeline 设计，因为 CPC 与 GAIC 的标注尺度不同。AAAI 论文中应把这个设计写成“heterogeneous supervision normalization by pairwise ranking”，而不是简单“混合数据集”。

### 4.4 VLM/Qwen middle state

VLM provider 的 prompt 只接收：

- image；
- known caption；
- initial semantic_type。

不接收 candidate boxes、MOS、best_crop。证据见 `composition_dataset_builder/vlm.py:392` 的 `_build_qwen_prompt`。

prompt 要求输出：

- caption；
- semantic_type；
- main_subject；
- key_objects；
- important_background；
- distractors；
- composition_intent；
- 可定位时输出 `bbox_norm`。

这能防守“teacher 是否看到裁剪答案”的质疑。但注意：`build_middle_state` 会把 `best_crop/best_score` 作为元数据写入记录。论文和代码注释要明确：这些字段不参与 VLM prompt，只用于后续数据记录和监督来源追踪。

### 4.5 RIG target construction

`build_rig_targets` 生成：

- nodes；
- relations；
- candidate_utilities；
- action_targets；
- graph_quality_flags。

关键代码：

- role/action/relation vocabulary：`RIGCrop/rigcrop/schema.py:10`。
- target 主入口：`RIGCrop/rigcrop/schema.py:44`。
- node extraction：`RIGCrop/rigcrop/schema.py:138`。
- relation construction：`RIGCrop/rigcrop/schema.py:234`。
- candidate utility construction：`RIGCrop/rigcrop/schema.py:277`。
- utility formula：`RIGCrop/rigcrop/schema.py:297`。

`candidate_utilities` 是 RIGCrop 的论文核心之一：它不是直接给 crop label，而是把 VLM graph 和 candidate crop 的几何关系转成可蒸馏的结构化辅助信号。

### 4.6 数据 pipeline 风险

| 风险 | 当前状态 | 建议 |
|---|---|---|
| VLM bbox 缺失率未知 | 本地无真实 audit JSON | 必须运行 schema audit 并报告 main_subject_bbox/key_objects_bbox |
| VLM hallucination | 当前仅有 prompt 约束 | 抽样人工验证 300-500 张，报告 teacher precision |
| candidate utility 可能和 crop label 不一致 | 当前无真实统计 | 计算 utility 与 human pairwise/MOS 的 Kendall/Spearman |
| CPC/GAIC 域差异 | 配置支持混训，但无结果 | 做 CPC->GAIC、GAIC->CPC cross-dataset generalization |
| label leakage 质疑 | provider prompt 未传标签，但元数据有 best_crop | 加 no-leakage audit 文档和单元测试 |

---

## 5. 训练与评估现状

### 5.1 训练目标

总 loss：

```text
L =
  crop_pair * L_crop_pair
  + node * L_node
  + relation * L_relation
  + utility * L_utility
  + action * L_action
```

代码证据：

- pairwise crop loss：`RIGCrop/rigcrop/losses.py`
- graph supervision：`RIGCrop/rigcrop/losses.py`
- utility distillation：`RIGCrop/rigcrop/losses.py`
- loss aggregation：`RIGCrop/scripts/train_rig_crop.py:123`

### 5.2 已有消融配置

已有配置：

- `RIGCrop/configs/ablations/rig_crop_cpc_crop_only.yaml`
- `RIGCrop/configs/ablations/rig_crop_cpc_no_importance.yaml`
- `RIGCrop/configs/ablations/rig_crop_cpc_no_relation.yaml`
- `RIGCrop/configs/ablations/rig_crop_cpc_no_utility.yaml`

这很重要。AAAI 论文必须把这些消融跑出来，否则创新点无法成立。

### 5.3 当前 smoke test

本次运行：

```bash
bash RIGCrop/scripts/run_smoke_test.sh
```

结果：

```text
train_pairs=12
val_pairs=6
train_pairwise_acc=0.9167
val_pairwise_acc=0.8333
weighted_pairwise_acc=0.8696
image-only predict: OK
```

解释：

- 说明端到端工程链路可执行；
- 不能作为方法有效性结果；
- 不应写入论文主实验表，只可作为开发自检。

### 5.4 AAAI 所需实验矩阵

最低主实验：

| 实验 | 数据集 | 指标 |
|---|---|---|
| CPC pairwise ranking | CPC val/test | pairwise acc、weighted acc、Kendall/Spearman |
| GAIC candidate ranking | GAICD test | top-1 MOS、top-k MOS、NDCG、rank correlation |
| top-k crop localization | CPC/GAIC | IoU@top1、Recall@IoU>0.75、best-of-k IoU |
| cross-dataset | train CPC eval GAIC、train GAIC eval CPC | 泛化性能 |
| image-only inference | 任意外部图像集 | latency、FPS、params、FLOPs |
| human preference | 至少 100-300 张 | pairwise human win-rate |

必须对比：

- GAIC-style ranker baseline；
- DACC CropRanker；
- DACCNet / query-based baseline；
- TransView 或公开可复现替代；
- S2CNet；
- Cropper/VLM teacher upper bound；
- crop-only RIGCrop；
- no relation / no importance / no utility。

---

## 6. 创新点评估

### 6.1 可成立的创新点

#### 创新点 1：VLM middle-state graph distillation

这点较强。RIGCrop 不是用 VLM 直接输出最终 crop，而是把 VLM 的构图分析压缩成结构化监督：

- entity role；
- importance；
- bbox；
- relation policy；
- action targets；
- crop-conditioned candidate utility。

如果实验证明这些监督提升 image-only student，创新点可以成立。

#### 创新点 2：显式 crop-conditioned relation utility

这点是当前代码最清晰的原创实现。`graph_features_for_crop` 将 predicted graph 和 candidate crop 结合，直接影响 score。

与已有 implicit relation methods 相比，RIGCrop 有一个更清楚的解释路径：

```text
Why is this crop better?
Because it preserves important nodes, preserves typed relations,
avoids cutting key entities, and removes distractors.
```

#### 创新点 3：heterogeneous supervision under pairwise formulation

CPC 和 GAICD 标注尺度不同。当前 pipeline 通过 pairwise 化处理异构监督，是合理且可写进方法部分的工程贡献。

#### 创新点 4：deployable image-only student

与 Cropper 这类 VLM inference 方法相比，RIGCrop 的部署叙事有价值：

- 不需要在线 VLM；
- 可批量处理；
- 成本低；
- 可在移动端/服务端轻量部署。

### 6.2 不足以单独构成创新的点

| 声称 | 为什么不够 |
|---|---|
| “我们使用 VLM 分析图像构图” | Cropper 等已有 VLM 裁剪方向 |
| “我们考虑主体和背景关系” | S2CNet、TransView 等已有关系建模 |
| “我们做 anchor crop scoring” | GAIC 已经成熟 |
| “我们用 pairwise preference 训练” | CPC 已经成熟 |
| “我们用弱监督构造裁剪数据” | GenCrop、ProCrop 已经非常接近 |

### 6.3 最推荐的论文题目方向

候选题目：

1. **RIGCrop: Distilling Relation-Importance Composition Graphs for Image-Only Cropping**
2. **Relation-Importance Guided Image Cropping via VLM Middle-State Distillation**
3. **Learning to Crop by Preserving What Matters: VLM-Distilled Composition Graphs for Image-Only Cropping**

---

## 7. Academic Paper Reviewer 模拟评审

### 7.1 Reviewer Configuration

| 角色 | 模拟身份 | 关注点 |
|---|---|---|
| EIC | AAAI computer vision / multimodal learning area editor | 是否有 AAAI 级别 novelty、是否超越工程整合 |
| R1 Methodology | 自动图像裁剪与视觉评测专家 | 数据、指标、基线、消融、统计可信度 |
| R2 Domain | image aesthetics / composition learning 专家 | 与 GAIC/CPC/TransView/S2CNet/Cropper/GenCrop/ProCrop 的关系 |
| R3 Perspective | 部署与多模态系统专家 | image-only student 的实用价值、成本、可靠性、伦理 |
| Devil's Advocate | 顶会审稿压力测试 | 找最强拒稿理由 |

### 7.2 EIC Review

**Recommendation：Major Revision / Borderline Reject**

**Strengths**

1. 主题与 AAAI 相关：结合 VLM teacher、结构化蒸馏、image-only student，有 AI 方法贡献潜力。
2. 方法主张比普通 crop ranker 更清楚：显式 relation-importance graph 参与 crop scoring。
3. 数据 pipeline 有工程成熟度：CPC/GAIC 转换、VLM enrichment、schema audit、RIG target 构建、消融配置齐全。

**Weaknesses**

1. 当前没有真实实验结果，不能支撑 AAAI 主会判断。
2. 方法架构仍偏 first version：TinyBackbone + fixed anchors，和近年 SOTA 差距可能较大。
3. novelty 需要小心定位，否则会被 S2CNet/Cropper/GenCrop/ProCrop 覆盖。

**EIC verdict**

如果论文现在投稿，EIC 大概率认为“promising but premature”。若补齐强实验和清晰叙事，可以进入 borderline-to-positive 区间。

### 7.3 Methodology Review

**Recommendation：Major Revision**

**主要问题**

1. 缺少真实训练/验证数据证据。仓库目前没有真实 `data/` JSONL 与训练 history。
2. 评估指标不足。pairwise accuracy 不能充分代表裁剪质量，必须补 MOS/NDCG/IoU/top-k/human preference。
3. VLM teacher 质量未量化。需要报告 middle-state coverage、bbox quality、relation/action label distribution、人审一致性。
4. 当前消融配置存在，但未见消融结果。创新点必须由 no-relation/no-utility/no-importance 结果支撑。
5. 统计显著性缺失。至少需要多 seed、confidence interval 或 bootstrap test。

**可修复性**

问题是可修复的，但需要完整实验 sprint。

### 7.4 Domain Review

**Recommendation：Major Revision**

**文献定位风险**

RIGCrop 必须正面比较：

- CPC/Good View Hunting：pairwise preference；
- GAIC/GAICD：grid anchor candidate scoring；
- TransView：crop boundary relation；
- S2CNet：spatial-semantic graph；
- GenCrop/ProCrop：弱监督/专业构图数据；
- Cropper：VLM-based cropping。

**贡献定位建议**

论文贡献不要写成：

> We introduce semantic relations for image cropping.

应写成：

> We use VLM-generated explicit middle states as structured supervision to train an image-only cropper whose scoring function depends on predicted relation-preservation utilities.

### 7.5 Perspective Review

**Recommendation：Minor-to-Major Revision**

**积极面**

从部署角度，RIGCrop 有价值。VLM 裁剪方法成本高、延迟高、不适合大规模或端侧部署。RIGCrop 把 VLM 理由蒸馏到小模型中，有实际意义。

**担忧**

1. VLM teacher 的 bias 会被 student 固化。
2. 自动裁剪可能误删少数群体、背景语境或文化符号。
3. 对“好构图”的定义可能过度依赖西方摄影规则或数据集偏好。

论文应增加：

- failure cases；
- culturally diverse examples；
- “no crop needed / fallback full image” 机制评估；
- 对敏感场景的限制说明。

### 7.6 Devil's Advocate Review

**Strongest Counter-Argument**

RIGCrop 当前最强反对意见是：它可能只是把现有 crop ranking、VLM pseudo-labeling 和启发式 graph features 拼在一起，而没有证明这种结构化中间态真的带来超越强视觉模型和强 VLM teacher 的泛化能力。已有工作已经覆盖 pairwise cropping、anchor scoring、relation-aware cropping、VLM cropping 和弱监督裁剪数据构造。若实验只显示相对小 CNN baseline 有提升，而不能击败或接近 S2CNet/Cropper/GenCrop/ProCrop 等强基线，AAAI 审稿人会认为贡献不足。

**CRITICAL**

| # | Issue | Location |
|---|---|---|
| C1 | 无真实实验结果，无法支撑任何性能主张 | 当前仓库状态 |
| C2 | novelty 叙事若写成“VLM + relation cropping”，会被已有工作覆盖 | 方法定位 |

**MAJOR**

| # | Issue | Location |
|---|---|---|
| M1 | fixed anchors + TinyBackbone 不足以代表顶会方法实力 | `RIGCrop/rigcrop/model.py` |
| M2 | teacher utility 公式启发式较强，缺少验证 | `RIGCrop/rigcrop/schema.py:297` |
| M3 | VLM bbox/relation/action 的质量未审计 | data pipeline |
| M4 | 缺少 human preference 和跨数据集泛化 | evaluation |

### 7.7 Editorial Synthesis

**综合决定：Major Revision before AAAI submission**

RIGCrop 具备一个可投稿的研究方向，但当前还处于“可执行原型 + 方法雏形”阶段。若目标是 AAAI 主会，必须先完成真实实验和强对比。若时间不足，应优先形成 workshop/arXiv 版本，避免主会投稿因为证据不足被直接拒掉。

---

## 8. AAAI 投稿路线图

### 8.1 AAAI-27 时间线

根据 AAAI-27 官方页面，当前重要日期为：

- 摘要截止：2026-07-21，UTC-12；
- 全文截止：2026-07-28，UTC-12；
- 补充材料和代码截止：2026-07-31，UTC-12；
- 作者反馈：2026-09-30 至 2026-10-06；
- 通知：2026-11-23。

截至 2026-06-23，全文截止前约 35 天。

### 8.2 最小可投稿补强包

如果要冲 AAAI-27，建议按优先级执行：

#### P0：真实数据跑通

- CPC -> DACC -> Qwen -> RIG targets；
- GAICD -> DACC -> Qwen -> RIG targets；
- 保存 summary/audit/history；
- 记录所有命令与随机种子。

#### P1：主模型与强 baseline

至少训练：

- CropRanker baseline；
- DACCNet baseline；
- RIGCrop crop-only；
- RIGCrop full；
- RIGCrop no-relation；
- RIGCrop no-utility；
- RIGCrop no-importance；
- 一个更强 backbone 版本。

#### P2：核心结果表

论文至少需要 4 张表：

1. Main comparison on CPC。
2. Main comparison on GAICD。
3. Ablation study。
4. Efficiency / deployment comparison。

还需要 2-3 张图：

1. 方法架构图。
2. graph + crop score 可解释可视化。
3. failure cases。

#### P3：人评

建议设计：

- 100-300 张图片；
- 每张比较 RIGCrop vs best baseline vs Cropper/VLM teacher；
- 3-5 名评审；
- 报告 win-rate、Fleiss/Krippendorff agreement。

#### P4：写作策略

论文结构：

1. Introduction：部署痛点 + VLM reasoning distillation。
2. Related Work：automatic cropping、relation-aware cropping、VLM cropping、weak supervision。
3. Method：VLM middle-state schema、RIG target construction、student architecture、losses。
4. Experiments：datasets、baselines、metrics、main results、ablation、人评、efficiency。
5. Analysis：teacher quality、explainability、failure cases。
6. Limitations：teacher bias、cultural composition bias、fixed anchors。

### 8.3 如果只能做 2 周 sprint

建议目标降级为：

- arXiv technical report；
- AAAI workshop；
- CVPR workshop；
- 项目 demo + dataset pipeline paper。

不要强行主会投稿，因为没有完整实验会让 reviewer 很难给高分。

---

## 9. 参考文献与链接

### 9.1 自动图像裁剪基础

1. Wei et al. **Good View Hunting: Learning Photo Composition from Dense View Pairs.** CVPR 2018.  
   https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Good_View_Hunting_CVPR_2018_paper.html

2. Li et al. **A2-RL: Aesthetics Aware Reinforcement Learning for Image Cropping.** CVPR 2018.  
   https://openaccess.thecvf.com/content_cvpr_2018/html/Li_A2-RL_Aesthetics_Aware_CVPR_2018_paper.html

3. Zeng et al. **Reliable and Efficient Image Cropping: A Grid Anchor Based Approach.** CVPR 2019.  
   https://openaccess.thecvf.com/content_CVPR_2019/html/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.html

4. Tu et al. **Image Cropping with Composition and Saliency Aware Aesthetic Score Map.** AAAI 2020.  
   https://ojs.aaai.org/index.php/AAAI/article/view/6889

5. Hong et al. **Composing Photos Like a Photographer.** CVPR 2021.  
   https://openaccess.thecvf.com/content/CVPR2021/html/Hong_Composing_Photos_Like_a_Photographer_CVPR_2021_paper.html

6. Pan et al. **TransView: Inside, Outside, and Across the Cropping View Boundaries.** ICCV 2021.  
   https://openaccess.thecvf.com/content/ICCV2021/html/Pan_TransView_Inside_Outside_and_Across_the_Cropping_View_Boundaries_ICCV_2021_paper.html

7. Jia et al. **Rethinking Image Cropping: Exploring Diverse Compositions from Global Views.** CVPR 2022.  
   https://openaccess.thecvf.com/content/CVPR2022/html/Jia_Rethinking_Image_Cropping_Exploring_Diverse_Compositions_From_Global_Views_CVPR_2022_paper.html

8. Wang et al. **Image Cropping with Spatial-Aware Feature and Rank Consistency.** CVPR 2023.  
   https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Image_Cropping_With_Spatial-Aware_Feature_and_Rank_Consistency_CVPR_2023_paper.html

### 9.2 近期强相关工作

9. Su et al. **Spatial-Semantic Collaborative Cropping for User Generated Content.** AAAI 2024.  
   https://ojs.aaai.org/index.php/AAAI/article/view/28303

10. Hong et al. **Learning Subject-Aware Cropping by Outpainting Professional Photos.** AAAI 2024.  
   https://ojs.aaai.org/index.php/AAAI/article/view/27990

11. Lee et al. **Cropper: Vision-Language Model for Image Cropping through In-Context Learning.** CVPR 2025.  
    https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Cropper_Vision-Language_Model_for_Image_Cropping_through_In-Context_Learning_CVPR_2025_paper.pdf

12. Zhao et al. **Can Machines Understand Composition? Dataset and Benchmark for Photographic Image Composition Embedding and Understanding.** CVPR 2025.  
    https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Can_Machines_Understand_Composition_Dataset_and_Benchmark_for_Photographic_Image_CVPR_2025_paper.html

13. Zhang et al. **ProCrop: Learning Aesthetic Image Cropping from Professional Compositions.** AAAI 2026.  
    https://ojs.aaai.org/index.php/AAAI/article/view/38255

### 9.3 AAAI 投稿信息

14. AAAI-27 official page.  
    https://aaai.org/conference/aaai/aaai-27/

15. AAAI-27 Main Technical Track Call.  
    https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/

---

## 10. 最终建议

RIGCrop 的想法值得继续，而且比普通 crop ranker 更有论文味道。真正需要警惕的是“讲法过大、证据不足”。当前最佳策略不是宣称已经达到 AAAI 水平，而是用 4-5 周把证据补成这样：

```text
RIGCrop full
  > crop-only
  > no-relation / no-utility / no-importance
  > DACC/GAIC ranker baselines
  comparable to or better than strong published methods on at least one setting
  much cheaper than VLM-inference Cropper
  explainable through graph/crop utility visualization
```

如果这条链跑通，AAAI 主会有可辩护的投稿价值。否则，建议先投 workshop/arXiv，继续升级为更强 backbone + query-based generator 后再冲主会。
