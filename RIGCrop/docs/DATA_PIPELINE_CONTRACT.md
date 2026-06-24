# RIGCrop 数据管线契约

本文档定义 RIGCrop 严格依赖的输入格式。RIGCrop 不直接解析 CPC 原始 `CollectedAnnotationsRaw/*.jpg.txt` 或 GAICD 原始 `.mat/.txt` 标注；这些原始格式必须先走现有 pipeline 转成 DACC-style JSONL。

## 1. 上游 pipeline

### CPC

```text
CPCDataset/
  CollectedAnnotationsRaw/*.jpg.txt
  images/*.jpg

scripts/cpc_to_dacc_jsonl.py
  -> data/cpc_dacc/metadata/train.jsonl
  -> data/cpc_dacc/metadata/val.jsonl

scripts/enrich_dacc_with_vlm_semantics.py --vlm qwen
  -> data/cpc_semantic_qwen/metadata/train.jsonl
  -> data/cpc_semantic_qwen/metadata/val.jsonl
```

CPC DACC 必需字段：

```text
sample_id
image_path
image_width
image_height
candidates[]
pairwise_preferences[]
cpc_supervision
```

每个 candidate 至少包含：

```text
candidate_id
box
scores.final_score 或 scores.cpc_raw_score
```

每个 pairwise preference 至少包含：

```text
winner
loser
weight
```

### GAICD

```text
GAICD raw
scripts/gaic_to_dacc_jsonl.py
  -> data/gaic_dacc/metadata/train.jsonl

scripts/enrich_dacc_with_vlm_semantics.py --vlm qwen
  -> data/gaic_semantic_qwen/metadata/train.jsonl
```

GAIC DACC 必需字段：

```text
sample_id
image_path
image_width
image_height
candidates[]
best_crop
best_score
gaic_supervision
```

每个 GAIC candidate 应包含：

```text
candidate_id
box
scores.mos 或 scores.final_score
```

RIGCrop 如果发现 GAIC 没有 `pairwise_preferences`，会从 candidate score 派生 pairwise ranking。它不会把 GAIC MOS 和 CPC pseudo score 直接混成一个统一回归标签。

## 2. Qwen middle state

RIGCrop 消费以下字段：

```text
composition_middle_state.main_subject
composition_middle_state.key_objects
composition_middle_state.important_background
composition_middle_state.distractors
composition_middle_state.composition_intent
```

推荐字段：

```text
name
category
description
importance
bbox_norm
relation_to_subject
location
```

`bbox_norm` 必须是：

```text
[x1, y1, x2, y2] in [0, 1]
```

如果 `bbox_norm` 缺失，RIGCrop 仍能训练 crop ranking、importance、action，但对应节点不会计算 bbox loss。

## 3. RIG targets 输出

`build_middle_state_targets.py` 会在每条记录中添加：

```text
rig_targets
```

结构：

```json
{
  "rig_targets": {
    "version": "rig_targets_v1",
    "source_middle_state": "qwen_dashscope",
    "crop_supervision_source": "cpc_pairwise_preference",
    "nodes": [],
    "relations": {},
    "candidate_utilities": {},
    "action_targets": {},
    "graph_quality_flags": {}
  }
}
```

### nodes

固定 padding 到 `max_nodes`。RIGFormer 生产默认是 12；smoke test 可显式使用 8：

```text
role: main_subject | key_object | important_background | distractor | padding
role_id
importance
bbox_norm
has_box
valid
```

### relations

```text
policy: [M, M]
weight: [M, M]
mask: [M, M]
```

relation policy：

```text
none
preserve_relation
optional_preserve
avoid_cutting
leave_space
distractor_exclusion
```

### candidate_utilities

按 `candidate_id` 存储：

```text
utility_raw
utility_unit
node_keep
relation_keep
boundary_cut_penalty
distractor_penalty
subject_position_score
box_feat
```

这些 utility 是从 Qwen graph 与 candidate crop 几何关系派生出来的辅助监督，不替代 CPC/GAIC 人工监督。

## 4. Schema audit

运行：

```bash
python RIGCrop/scripts/audit_middle_state_schema.py \
  --jsonl data/cpc_semantic_qwen/metadata/train.jsonl \
  --out-json data/cpc_rig/metadata/train.audit.json
```

重点看：

```text
rates.middle_state
rates.main_subject_bbox
rates.key_objects_bbox
top_relation_to_subject
top_suggested_actions
```

如果 `main_subject_bbox` 很低，训练时应降低 node bbox loss 权重，或先改 Qwen prompt 补 bbox。
