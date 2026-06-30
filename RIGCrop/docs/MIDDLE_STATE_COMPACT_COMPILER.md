# Composition Middle-State Compact Compiler

日期：2026-06-30  
目标：把 VLM 生成的丰富构图解释压缩成低熵、固定 schema、可训练的 composition middle-state，避免训练数据过散、JSONL 过大、DataLoader 过慢，同时保留论文里“中间态蒸馏”的主线。

---

## 1. 为什么需要 compact compiler

VLM/Qwen 可以生成很多有解释性的字段：

```text
name / category / description / relation_to_subject
reason / critique / failure_modes / free-form composition explanation
```

这些内容对人工审计和论文可视化有价值，但不应该直接进入训练主文件。训练真正需要的是稳定的结构化监督：

```text
node role
node bbox
node importance
relation policy
relation weight
action multi-hot
candidate utility scalars
human crop preference / MOS-derived pairwise preference
```

因此当前 pipeline 明确分成两层：

```text
raw middle-state JSONL
  -> compact compiler
  -> train middle-state JSONL
  -> RIGPairwiseDataset
  -> RIGFormer
```

这样可以让大模型“想得多”，但让训练只学“压缩后的稳定结构”。

---

## 2. 当前代码改动

### 2.1 Schema-level compiler

文件：

```text
RIGCrop/rigcrop/schema.py
```

新增函数：

```python
compact_rig_record(...)
compact_rig_targets(...)
```

作用：

- 保留 DACC/RIG 训练必需字段。
- 删除 raw `composition_middle_state` / `vlm_understanding`，除非显式要求保留。
- 删除 `rig_targets.nodes` 中的 `name/category/description/relation_to_subject`，除非显式要求保留。
- 保留 `candidate_utilities` 中的数值分量，但删除可重算或审计型冗余字段。

### 2.2 Dataset memory compiler

文件：

```text
RIGCrop/rigcrop/data.py
```

`RIGPairwiseDataset` 新增参数：

```yaml
compact_records: true
keep_raw_middle_state: false
keep_node_text: false
```

默认行为：

```text
load_jsonl()
  -> compact_rig_record()
  -> build candidate map
  -> train samples
```

也就是说，即使你暂时还没离线生成 compact JSONL，训练进程读入内存后也会丢掉不参与训练的长文本字段。

### 2.3 Offline compact JSONL script

新增脚本：

```text
RIGCrop/scripts/compile_middle_state_train_jsonl.py
```

这个脚本用于把原始 JSONL 编译成训练版 JSONL，并输出压缩率和 audit summary。

---

## 3. Compact JSONL 保留什么

### 3.1 Top-level fields

训练版 JSONL 保留：

```text
sample_id
image_path
rel_path
image_width / image_height
best_crop / best_score
candidates
pairwise_preferences
cpc_supervision / gaic_supervision
quality_flags
rig_targets
```

默认删除：

```text
composition_middle_state
vlm_understanding
raw VLM descriptions
free-form reasons
prompt artifacts
```

### 3.2 Candidate fields

每个 candidate 保留：

```text
candidate_id
box
score
scores.mos
scores.final_score
scores.cpc_raw_score
scores.aesthetic_score
scores.technical_score
scores.composition_score
```

这些足够支持：

- CPC explicit pairwise preference
- GAIC MOS-derived pairwise preference
- candidate ranking evaluation

### 3.3 RIG target fields

每个 node 默认只保留：

```text
node_id
role
role_id
importance
bbox_norm
has_box
valid
```

relation 保留：

```text
policy
weight
mask
```

candidate utility 保留：

```text
utility_raw
utility_unit
node_keep
relation_keep
boundary_cut_penalty
distractor_penalty
subject_position_score
```

action target 只保留：

```text
multi_hot
```

---

## 4. 推荐数据目录

不要覆盖 raw 数据，建议分层保存：

```text
data/cpc_rig/metadata/train.jsonl              # raw / audit-rich RIG JSONL
data/cpc_rig/metadata/train.compact.jsonl      # compact train JSONL
data/cpc_rig/metadata/val.compact.jsonl

data/gaic_rig/metadata/train.jsonl
data/gaic_rig/metadata/train.compact.jsonl
data/gaic_rig/metadata/val.compact.jsonl
```

raw 文件用于：

```text
人工审计
可视化
prompt debug
中间态案例展示
```

compact 文件用于：

```text
训练
评估
多 worker DataLoader
跨服务器迁移
```

---

## 5. 生成 compact JSONL

### CPC train

```bash
cd /home/hmx/workspace/imageCut

python RIGCrop/scripts/compile_middle_state_train_jsonl.py \
  --input-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl \
  --out-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.compact.jsonl \
  --max-nodes 12 \
  --overwrite
```

### CPC val

```bash
python RIGCrop/scripts/compile_middle_state_train_jsonl.py \
  --input-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl \
  --out-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.compact.jsonl \
  --max-nodes 12 \
  --overwrite
```

### GAIC train

```bash
python RIGCrop/scripts/compile_middle_state_train_jsonl.py \
  --input-jsonl /home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.jsonl \
  --out-jsonl /home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.compact.jsonl \
  --max-nodes 12 \
  --overwrite
```

### GAIC val

```bash
python RIGCrop/scripts/compile_middle_state_train_jsonl.py \
  --input-jsonl /home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/val.jsonl \
  --out-jsonl /home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/val.compact.jsonl \
  --max-nodes 12 \
  --overwrite
```

脚本会输出：

```text
*.compact.jsonl.summary.json
```

重点看：

```text
processed
input_bytes
output_bytes
size_reduction_ratio
audit_first_records.rates
```

---

## 6. 如果输入还没有 rig_targets

脚本默认会在缺少 `rig_targets` 时自动构建。  
如果你想强制重建：

```bash
python RIGCrop/scripts/compile_middle_state_train_jsonl.py \
  --input-jsonl /path/to/raw_semantic_qwen.jsonl \
  --out-jsonl /path/to/train.compact.jsonl \
  --max-nodes 12 \
  --rebuild-rig-targets \
  --overwrite
```

---

## 7. 如果需要保留文本用于审计

默认不保留 raw middle-state 和 node text。  
如果要生成可审计但仍较紧凑的版本：

```bash
python RIGCrop/scripts/compile_middle_state_train_jsonl.py \
  --input-jsonl /path/to/train.jsonl \
  --out-jsonl /path/to/train.audit_compact.jsonl \
  --max-nodes 12 \
  --keep-node-text \
  --overwrite
```

如果确实要保留完整 raw middle-state：

```bash
python RIGCrop/scripts/compile_middle_state_train_jsonl.py \
  --input-jsonl /path/to/train.jsonl \
  --out-jsonl /path/to/train.raw_kept.jsonl \
  --max-nodes 12 \
  --keep-raw-middle-state \
  --keep-node-text \
  --overwrite
```

训练不推荐这个版本。

---

## 8. 训练配置如何切到 compact JSONL

主配置可以把路径改成：

```yaml
train_datasets:
  - jsonl_path: /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.compact.jsonl
    image_size: 384
    crop_size: 224
    max_pairs_per_record: 128
    max_nodes: 12
    derive_pairs_from_scores: true
    image_cache_size: 8
    compact_records: true

  - jsonl_path: /home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.compact.jsonl
    image_size: 384
    crop_size: 224
    max_pairs_per_record: 128
    max_nodes: 12
    derive_pairs_from_scores: true
    min_score_gap: 0.05
    image_cache_size: 8
    compact_records: true

val_dataset:
  jsonl_path: /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.compact.jsonl
  image_size: 384
  crop_size: 224
  max_pairs_per_record: 128
  max_nodes: 12
  derive_pairs_from_scores: true
  image_cache_size: 8
  compact_records: true
```

`compact_records: true` 是默认值；写出来只是为了配置可读。

---

## 9. 验证 compact JSONL 是否可训练

```bash
PYTHONPATH=$PWD/RIGCrop python - <<'PY'
from torch.utils.data import DataLoader
from rigcrop.data import RIGPairwiseDataset

ds = RIGPairwiseDataset(
    jsonl_path="/home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.compact.jsonl",
    image_size=384,
    crop_size=224,
    max_records=16,
    max_pairs_per_record=8,
    max_nodes=12,
)
print("pairs", len(ds))
batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
for key in ["image", "winner_crop", "loser_crop", "node_boxes", "relation_policy", "action_targets"]:
    print(key, batch[key].shape)
print("sample_id", batch["sample_id"][:2])
PY
```

如果输出 tensor shape 正常，说明 compact 文件没有破坏训练输入。

---

## 10. 论文叙事如何使用

这部分可以写成方法小节：

```text
Schema-Constrained Middle-State Compiler.
Raw VLM reasoning is expressive but noisy and high-entropy. We therefore
compile it into a compact, schema-constrained composition middle-state
consisting only of trainable categorical and scalar variables. Free-form
textual rationales are retained for audit but excluded from the training JSONL.
```

建议强调：

- 中间态不是直接自由文本监督。
- 中间态经过 ontology mapping、top-K node selection、relation policy normalization、utility scalarization。
- 训练只学习 compact middle-state，避免被 VLM 长文本噪声拖散。

---

## 11. v2 中间态扩展原则

如果后续做 `composition_middle_state_v2`，不要把所有 VLM 输出直接塞进训练文件。仍然遵守：

```text
raw VLM output
  -> normalize
  -> map to fixed ontology
  -> filter low-confidence labels
  -> top-K nodes / relations
  -> compact train JSONL
```

推荐上限：

```text
nodes: <= 12
relations: <= 24
relation types: 6-8
crop utility components: 5-6
action labels: <= 16, per-image active labels 1-4
```

v2 可以增加：

```text
relation confidence
relation soft label
crop utility confidence
VLM multi-sample consistency
```

但训练文件仍然应该是 compact numeric/categorical schema，而不是长文本 reasoning dump。

---

## 12. 预期收益

工程收益：

- 降低 JSONL 文件体积。
- 降低 DataLoader 内存占用。
- 降低多 worker 重复加载成本。
- 减少训练端无用字段传播。

方法收益：

- 让“中间态蒸馏”从 raw VLM pseudo-label 变成 schema-constrained supervision framework。
- 回应审稿人对中间态冗余、噪声、不可学习的质疑。
- 支持论文中比较 `raw/rule-based middle-state` 与 `compact trainable middle-state` 的消融。
