# 图像裁剪方向 CVPR/AAAI 论文选题调研

本文档基于 `academic-research-suite` 的 deep-research 工作流整理，目标是：充分调研现有图像裁剪论文，并结合当前项目的数据构造 pipeline，提出具备 CVPR / AAAI 投稿潜力的研究方向。

## 1. 研究问题收敛

当前工程已经具备一个离线 teacher 数据构造 pipeline：

```text
原图
  -> VLM/Qwen 结构化理解
  -> YOLO / Grounding / VLM bbox 定位主体和关键物
  -> SAM / bbox mask 获取主体、关系物、背景、干扰物区域
  -> crop-state graph
  -> 方向性策略生成 candidates
  -> GAIC-style candidates
  -> 规则/美学/VLM 多维评分排序
  -> JSONL: masks, candidates, scores, action, reason, best_crop
```

要达到 CVPR / AAAI 论文标准，不能只把它写成“用 Qwen + SAM 造数据”。更有潜力的研究问题是：

> Can image cropping be improved by learning explicit composition-correction directions, rather than only predicting or ranking crop boxes?

中文表达：

> 图像裁剪是否应该从“预测一个好框”升级为“理解当前构图问题，并学习裁剪框应该如何移动、缩放和保留主体关系”？

这能形成一个明确的新任务：

```text
Direction-Aware Image Composition Correction
```

或者：

```text
Direction-Aware Image Cropping
```

## 2. 现有图像裁剪研究脉络

### 2.1 传统候选框评分与 ranking

早期深度裁剪方法多把裁剪看作候选框评价问题：

- VFN/VEN/VPN 通过 pairwise ranking 或 teacher-student 方式学习哪个 view 更好。CPC / Good View Hunting 构造了大规模比较视图对，强调 view selection 的比较性质。
- GAIC 进一步指出图像裁剪具有非唯一性，单一 ground-truth 和 IoU 指标不可靠，因此提出 grid anchor formulation 和 GAICD dense MOS 标注。

代表：

- Good View Hunting / CPC：提出 Comparative Photo Composition dataset，包含超过 100 万比较视图对。
- GAIC：每图用 grid anchor 产生有限候选，对所有候选打 MOS 分数；评价用 SRCC 和 AccK/N，而不是只用 IoU。

与本项目的关系：

```text
你的 candidates + scores + ranks 设计，本质上继承 GAIC/CPC 的 ranking 思路；
但你多了主体 mask、关系区域、构图问题、方向性 action/reason。
```

### 2.2 构图规则显式建模

CVPR 2021 的 Composing Photos Like a Photographer 认为已有方法大多隐式学习摄影知识，提出通过 key composition map 显式编码构图规则。

这个方向说明：

```text
摄影构图规则可以作为可学习信号，而不是只让网络从 bbox score 中隐式猜。
```

与本项目的关系：

```text
你的 action/reason 标签可以看作更高层的构图规则监督：
move_left, zoom_out, preserve_relation, remove_distractor, keep_environment。
```

### 2.3 多样 crop set prediction

CVPR 2022 Rethinking Image Cropping 指出：

```text
anchor evaluation 有多样性但缺少全局性；
coordinate regression 有全局性但通常只输出一个 crop；
更好的方式是从 global views 输出 diverse crops。
```

这为你的 top-k crop generator 提供直接理论支撑。

你的 pipeline 应避免：

```text
image -> one box
```

而应训练：

```text
image -> {box_i, score_i, action_i}_{i=1..K}
```

### 2.4 Spatial-aware / RoI + RoD / Rank consistency

CVPR 2023 Image Cropping with Spatial-Aware Feature and Rank Consistency 继续强化：

```text
crop 评价不仅看 RoI，还要看被丢弃区域 RoD；
裁剪排序应保持 rank consistency。
```

与本项目的关系：

```text
你可以把 RoD 从纯视觉区域升级成语义 RoD：
被裁掉的是干扰物、空白区，还是主体手脚/关键物/重要背景？
```

这是一个很好的论文切入点。

### 2.5 Human-centric / Subject-aware cropping

ECCV 2022 Human-centric Image Cropping 明确利用 human bbox，将图像按人体位置分区，并提出 content-preserving feature。

AAAI 2024 GenCrop 用 professional stock photos + diffusion outpainting 构造 cropped/uncropped 对，无需人工标注即可学习 subject-aware cropping。

与本项目的关系：

```text
GenCrop 是“好图 -> outpainting 扩图 -> 学回好 crop”；
你的 pipeline 是“好/坏构图 -> VLM/SAM/规则诊断 -> 学习方向性构图修正”。
```

这两者相邻，但你的关键差异应是：

```text
不仅生成 cropped-uncropped pair，
还生成 issue/action/reason/crop-state graph/ranked candidates。
```

### 2.6 VLM / MLLM cropping

CVPR 2025 Cropper 证明 VLM 可以通过 in-context learning 做 free-form、subject-aware、aspect-ratio-aware cropping。

ACM MM 2025 InstructCrop 进一步把 MLLM 教成 aesthetic cropper，并生成裁剪解释。

2026 arXiv CROP 开始把 aesthetic cropping 视为 multimodal reasoning，采用 analysis-proposal-decision 和 preference alignment。

这些工作说明：

```text
VLM reasoning for cropping 已经是显著趋势。
```

因此你不能把“Qwen 能理解图像并给裁剪建议”作为唯一贡献。更稳的贡献是：

```text
把 VLM reasoning 蒸馏为大规模结构化监督，
训练一个快速、端到端、方向感知的裁剪模型。
```

## 3. 文献矩阵

| 方向 | 代表工作 | 核心思想 | 局限 | 你的机会 |
|---|---|---|---|---|
| Pairwise ranking | VFN/VEN/VPN, CPC | 学习 view preference | 缺少语义方向，依赖候选或比较对 | 用 action/reason 扩展 preference |
| Dense candidate scoring | GAIC | grid anchor + 全候选 MOS | 候选有限，语义/主体关系弱 | 保留 candidates + scores，但加入 mask/scene graph |
| Composition rules | Composing Photos Like a Photographer | 显式构图图 / composition map | 规则层级较低 | 提升为 issue/action/reason |
| Diverse crop prediction | Rethinking Image Cropping | set prediction 输出多样 crops | 缺少方向性解释 | 输出 top-k box + action + issue |
| Spatial-aware ranking | Spatial-aware Feature and Rank Consistency | RoI/RoD + rank consistency | RoD 语义不足 | 语义 RoD：裁掉的是谁 |
| Human-centric | Human-centric Image Cropping | human bbox 分区、内容保护 | 主要是单人/人像 | 扩展到人-物、人-环境、多主体关系 |
| Weak supervision | GenCrop, ProCrop | professional photos + outpainting | 数据是 cropped/uncropped pair，缺少构图诊断 | 加入构图问题和方向标签 |
| VLM cropping | Cropper, InstructCrop, CROP | VLM reasoning / ICL / explanation | 推理慢、难部署、稳定性和成本问题 | VLM 作为离线 teacher，student 快速推理 |

## 4. 核心研究缺口

### Gap 1: 缺少方向性构图监督

多数工作输出：

```text
crop box / crop score
```

很少输出：

```text
为什么裁
当前构图问题是什么
该往哪移动
该 zoom in 还是 zoom out
该保留谁和谁的关系
```

你的 `action + issue + reason` 正好补这个缺口。

### Gap 2: 缺少语义级 RoD 评价

已有 RoD 通常是视觉特征层面。你的 pipeline 能判断：

```text
裁掉的是干扰物 -> 好
裁掉的是主体手脚 -> 坏
裁掉的是关键物 -> 坏
裁掉的是可选背景 -> 视场景而定
```

这是比普通 RoI/RoD 更强的语义监督。

### Gap 3: VLM cropping 缺少高效蒸馏

VLM cropper 的优势是推理和审美 reasoning，劣势是慢、贵、难控。

你的方向可以是：

```text
VLM/SAM/rule teacher -> structured supervision -> lightweight student
```

这比直接用 VLM 生成 crop 更适合产品和高分辨率批处理。

### Gap 4: 现有数据集缺少 mask/relation/action 层标签

GAICD 有 dense MOS。
CPC 有比较对。
GenCrop/ProCrop 有弱监督 cropped-uncropped。

但缺：

```text
subject mask
key-object mask
relationship region
composition issue
directional action
candidate reason
```

如果你能公开一个中等规模 benchmark，这会是实质贡献。

## 5. 最推荐论文方向一：DACC

### 标题候选

```text
DACC: Direction-Aware Composition Correction for Image Cropping
```

或者：

```text
Learning Direction-Aware Image Cropping from Multimodal Teacher Signals
```

### 核心贡献

1. 提出 Direction-Aware Image Cropping 新任务。
2. 构建包含 `crop_state_graph + issue/action/reason + ranked candidates` 的数据集。
3. 提出端到端 Direction-aware Crop Transformer，输出 top-k crops、scores、action 和 issue。
4. 在 GAICD/CPC/FCDB/SACD/自建 benchmark 上验证，并做人类偏好评估。

### 方法设计

Teacher 数据构造：

```text
Qwen-VL: 主体、关系、构图意图、action/reason
YOLO/Grounding: bbox 定位
SAM: mask
规则系统: direction candidates
GAIC: grid candidates
Scorer: subject integrity + relation preservation + distractor removal + aesthetic/VLM preference
```

Student 模型：

```text
Image backbone: ConvNeXt / Swin / ViT / VMamba
Crop decoder: DETR-style K queries
Heads:
  box head
  score head
  action head
  issue head
  optional relation-preservation head
```

训练损失：

```text
L = L_box
  + L_rank
  + L_action
  + L_issue
  + L_subject_preserve
  + L_relation_preserve
  + L_teacher_distill
```

### 为什么能冲会

CVPR 角度：

```text
新视觉任务定义 + 数据集 + 模型 + benchmark。
```

AAAI 角度：

```text
VLM teacher reasoning 蒸馏 + human preference alignment + 可解释裁剪决策。
```

### 主要风险

如果没有强 student 模型，只展示 pipeline，会被认为是工程拼装。

必须证明：

```text
student 在不调用 Qwen/SAM 的情况下，也能超过已有方法；
direction/action 监督确实带来增益；
human preference 认为结果更好。
```

## 6. 论文方向二：Semantic RoD Cropping

### 标题候选

```text
Semantic Region-of-Discard Reasoning for Image Cropping
```

### 核心问题

已有 RoI/RoD 方法知道“保留区域”和“丢弃区域”，但不知道丢弃区域的语义价值。

你的方法可以定义：

```text
Good discard:
  empty region, border clutter, distractor

Bad discard:
  subject part, key object, relation region, important environment
```

### 方法

1. 用 VLM/SAM 构建 semantic RoD label。
2. 对每个候选 crop 计算：

```text
semantic_keep_score
semantic_discard_score
relation_cut_penalty
distractor_removal_score
```

3. 训练一个 Semantic-RoD Ranker。

### 优点

这个方向更聚焦，容易写出清晰 novelty：

```text
从视觉 RoD 到语义 RoD。
```

### 风险

如果只做 ranker，不做端到端 crop generator，CVPR 主会可能略弱；AAAI 或 ACM MM 更合适。

## 7. 论文方向三：Counterfactual Composition Learning

### 标题候选

```text
Counterfactual Composition Learning for Image Cropping
```

### 核心想法

借鉴 InstantRetouch / GenCrop：

```text
好构图 crop*
  -> 人为制造坏构图 crop_bad
  -> 标注 issue/action
  -> 学习从 bad composition 到 good composition 的 correction vector
```

坏构图生成：

```text
subject too left/right
headroom too tight
feet cut
key object missing
too much empty region
important background cut
wrong aspect ratio
```

模型输出：

```text
delta box / action / corrected crop
```

### 优点

这个方向很像“图像裁剪版 instruction editing”，有清晰因果/反事实味道。

### 风险

合成坏构图和真实坏构图有 domain gap。必须加入真实用户图或人工采样验证。

## 8. 论文方向四：Relation-Aware Subject Cropping

### 标题候选

```text
Relation-Aware Subject Cropping with Multimodal Composition Graphs
```

### 核心问题

Subject-aware cropping 通常关注主体是否完整，但很多真实裁剪需要保留关系：

```text
人和相机
人和宠物
人和车
人和建筑
人和动作方向
商品和使用场景
```

### 方法

构建 composition graph：

```text
nodes: subject, key object, environment, distractor
edges: holding, riding, facing, standing near, looking toward
```

训练 graph-aware crop decoder：

```text
image features + graph tokens -> top-k relation-preserving crops
```

### 优点

如果你的人-物/人-环境规则已经多，这个方向最贴合当前工程。

### 风险

需要足够多关系类型和可靠标注，否则容易被认为只是 human-centric cropping 的扩展。

## 9. 论文方向优先级

| 优先级 | 方向 | CVPR 潜力 | AAAI 潜力 | 推荐理由 |
---|---|---:|---:|---|
| 1 | DACC: Direction-Aware Composition Correction | 高 | 高 | 新任务 + 数据 + 模型，最完整 |
| 2 | Semantic RoD Reasoning | 中 | 高 | 概念清晰，适合 reasoning/ranking |
| 3 | Counterfactual Composition Learning | 中高 | 中高 | 数据构造有亮点，但需处理 domain gap |
| 4 | Relation-Aware Subject Cropping | 中 | 中高 | 工程贴合，但需证明超越 human-centric |

最建议主攻：

```text
DACC = Direction-aware task + semantic RoD + counterfactual data
```

把方向二、三作为 DACC 的子模块和 ablation。

## 10. 论文主线建议

### Abstract 主线

```text
Image cropping is usually formulated as crop-box regression or candidate ranking.
However, high-quality cropping often requires directional composition correction:
moving the frame, zooming in/out, preserving subject-object relations, and discarding distracting regions.
We introduce Direction-Aware Image Cropping, a new formulation that supervises not only where to crop, but why and how to adjust the crop.
```

### Introduction 论证链

```text
1. 裁剪是多解问题，GAIC/CPC 已证明单一框监督不足。
2. 近年方法考虑多样 crop、spatial-aware ranking、human-centric subject preservation。
3. VLM cropper 证明 multimodal reasoning 有用，但推理慢且难部署。
4. 现有方法缺少“方向性构图修正”监督。
5. 我们提出 DACC：VLM/SAM/规则 teacher 生成结构化构图标签，并蒸馏到端到端 student。
```

### Contributions

建议写成：

```text
1. We introduce direction-aware image cropping, where each crop is supervised with composition issue, corrective action, and ranked candidate quality.
2. We construct a multimodal teacher pipeline that converts raw images into crop-state graphs with subject masks, relation regions, semantic discard regions, and action-labeled crop candidates.
3. We propose a lightweight crop generator/ranker that distills teacher reasoning into real-time top-k crop prediction.
4. Extensive experiments on public cropping benchmarks and a new direction-aware benchmark show improved crop quality, subject preservation, relation preservation, and human preference.
```

## 11. 必做实验

### 11.1 数据集

公开数据：

```text
GAICD / GAICv1 / GAICv2
CPC
FCDB
FLMS
SACD / human-centric subsets
```

自建数据：

```text
DACC-Data
  images: 50K-200K
  candidates per image: 30-100
  labels: score, rank, issue, action, reason, masks, graph
```

人工验证子集：

```text
1K-3K images
3-5 annotators per image
top-k preference / action correctness / subject preservation
```

### 11.2 Baselines

必须比较：

```text
GAIC
VEN/VPN 或 CPC-based ranker
Rethinking Image Cropping / set prediction 方法
Spatial-aware Feature + Rank Consistency
Human-centric Image Cropping
GenCrop
ProCrop
Cropper / InstructCrop / VLM zero-shot 或 in-context cropping
你的旧规则系统
```

### 11.3 指标

通用 crop 指标：

```text
SRCC
AccK/N
IoU
Boundary displacement / Disp
top-k recall
```

你需要新增指标：

```text
Subject Preservation Rate
Key Object Preservation Rate
Relation Preservation Rate
Semantic Discard Accuracy
Action Prediction Accuracy
Issue Prediction Accuracy
Human Preference Win Rate
Inference latency
```

### 11.4 Ablation

必须做：

```text
w/o VLM understanding
w/o SAM/mask
w/o direction action loss
w/o semantic RoD scoring
w/o counterfactual bad crops
w/o existing rule candidates
w/o rank loss
teacher-only vs student
rule-only vs learned model
Qwen teacher vs heuristic teacher
```

## 12. AAAI / CVPR 投稿策略

### 更像 CVPR 的版本

重点放在：

```text
新视觉任务
公开 benchmark
端到端视觉模型
大量 public benchmark SOTA
速度/泛化/鲁棒性
```

标题风格：

```text
Direction-Aware Image Cropping via Composition State Distillation
```

### 更像 AAAI 的版本

重点放在：

```text
VLM reasoning distillation
preference alignment
interpretable action/reason
human-in-the-loop / AI reasoning
```

标题风格：

```text
Learning to Crop by Reasoning About Composition Corrections
```

## 13. Devil's Advocate 审查

### 最强反对意见

这项工作可能被批评为：

```text
只是把 Qwen、SAM、YOLO、规则和 GAIC 拼起来，缺少真正算法贡献。
```

### 必须回应

论文必须证明：

```text
1. direction-aware supervision 是必要的；
2. student 学到了 teacher 的可泛化知识，而不是复制规则；
3. 模型在公开数据集和人工偏好上优于已有方法；
4. 不是只在自建数据上好；
5. 不依赖测试时调用 Qwen/SAM。
```

### 高风险同步工作

Cropper、InstructCrop、CROP、ProCrop 已经很接近 VLM/aesthetic reasoning cropping。你的差异必须写清楚：

```text
他们：VLM 直接推理 / professional weak supervision / 多 crop proposal。
你：结构化方向性监督 + semantic RoD + action-labeled ranked candidates + student distillation。
```

## 14. 最小可投稿版本

如果目标是 AAAI：

```text
数据集: 50K images + 1K human-validated subset
模型: direction-aware crop transformer
实验: public benchmarks + human preference + ablation
贡献: VLM reasoning distillation + action/explanation supervision
```

如果目标是 CVPR：

```text
数据集: 100K-200K images + public release
模型: strong end-to-end crop set predictor
实验: 5+ benchmarks, SOTA comparison, efficiency
贡献: new task + dataset + model + metrics
```

## 15. 推荐最终方案

建议主线定为：

```text
DACC: Direction-Aware Composition Correction for Image Cropping
```

核心 slogan：

```text
Cropping is not only where to crop, but how to correct composition.
```

数据贡献：

```text
DACC-Data: first dataset with crop-state graph, semantic masks, issue/action labels, ranked crop candidates.
```

模型贡献：

```text
DACCNet: a student crop generator that distills multimodal teacher reasoning into real-time top-k crop prediction.
```

评价贡献：

```text
Beyond IoU/MOS: subject preservation, relation preservation, semantic discard, action correctness, human preference.
```

## 16. 关键参考资料

- Zeng et al., "Reliable and Efficient Image Cropping: A Grid Anchor based Approach", CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.pdf
- Wei et al., "Good View Hunting: Learning Photo Composition from Dense View Pairs", CVPR 2018. https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Good_View_Hunting_CVPR_2018_paper.pdf
- Hong et al., "Composing Photos Like a Photographer", CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Hong_Composing_Photos_Like_a_Photographer_CVPR_2021_paper.html
- Jia et al., "Rethinking Image Cropping: Exploring Diverse Compositions From Global Views", CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Jia_Rethinking_Image_Cropping_Exploring_Diverse_Compositions_From_Global_Views_CVPR_2022_paper.html
- Zhang et al., "Human-centric Image Cropping with Partition-aware and Content-preserving Features", ECCV 2022. https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670176.pdf
- Wang et al., "Image Cropping With Spatial-Aware Feature and Rank Consistency", CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Image_Cropping_With_Spatial-Aware_Feature_and_Rank_Consistency_CVPR_2023_paper.pdf
- Hong et al., "Learning Subject-Aware Cropping by Outpainting Professional Photos", AAAI 2024 / GenCrop. https://arxiv.org/abs/2312.12080
- Lee et al., "Cropper: Vision-Language Model for Image Cropping through In-Context Learning", CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Cropper_Vision-Language_Model_for_Image_Cropping_through_In-Context_Learning_CVPR_2025_paper.pdf
- Sheng et al., "InstructCrop: Teaching Multimodal Large Language Models to Crop Aesthetic Images", ACM MM 2025. https://dl.acm.org/doi/10.1145/3746027.3754931
- "ProCrop: Learning Aesthetic Image Cropping from Professional Compositions", AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/38255
- Dong et al., "CROP: Expert-Aligned Image Cropping via Compositional Reasoning and Optimizing Preference", arXiv 2026. https://arxiv.org/abs/2605.12545
- Kirillov et al., "Segment Anything", ICCV 2023. https://arxiv.org/abs/2304.02643
- Qwen2.5-VL official blog: visual grounding and JSON output. https://qwenlm.github.io/blog/qwen2.5-vl/
- AAAI-26 Main Technical Track Call. https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/
- CVPR 2026 Call for Papers. https://cvpr.thecvf.com/Conferences/2026/CallForPapers

