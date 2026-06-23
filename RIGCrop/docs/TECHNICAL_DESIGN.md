# RIGCrop 技术设计

## 1. 目标

训练一个 image-only crop student：

```text
input: image
output: top-k crop boxes + scores
```

训练时使用：

```text
CPC pairwise preferences
GAIC MOS / MOS-derived pairs
Qwen/VLM middle-state graph targets
```

推理时不使用：

```text
Qwen
caption
middle-state JSON
detector
SAM
dataset candidates
```

## 2. 创新点落地

RIGCrop 的重点不是“训练时多一个 caption”，而是：

```text
Qwen middle state
  -> nodes: role, importance, bbox
  -> relations: preserve / avoid_cutting / leave_space / distractor_exclusion
  -> candidate utility: crop-conditioned relation preservation
```

模型学习一个 latent composition graph，然后用这个图参与 crop scoring。

## 3. 模型结构

文件：

```text
RIGCrop/rigcrop/model.py
```

主要类：

```python
RIGCropModel
```

结构：

```text
image
  -> TinyBackbone
  -> node tokens
      -> node box
      -> node role
      -> node importance
      -> node valid
      -> relation policy
      -> relation weight
  -> candidate crop branch
  -> graph_features_for_crop
  -> score_head
```

`graph_features_for_crop` 包含：

```text
node_keep
relation_keep
distractor
boundary
main_cov
```

这些特征使 crop score 显式依赖 predicted graph，而不是只依赖 crop patch。

## 4. Loss

文件：

```text
RIGCrop/rigcrop/losses.py
```

总 loss：

```text
L =
  crop_pair * L_crop_pair
  + node * L_node
  + relation * L_relation
  + utility * L_utility
  + action * L_action
```

### Crop pair loss

```text
softplus(-(score(winner) - score(loser))) * weight
```

CPC 使用原始 `pairwise_preferences`。

GAIC 如果没有 pairwise，则从 candidate MOS/final_score 派生 pair。

### Node loss

```text
SmoothL1 node bbox
CE node role
SmoothL1 importance
BCE valid
```

无 bbox 的 teacher node 不计算 bbox loss。

### Relation loss

```text
CE relation policy
SmoothL1 relation weight
```

只在 valid relation mask 上计算。

### Utility distillation

```text
SmoothL1(pred utility, teacher utility)
+ teacher-utility-derived pairwise utility ranking
```

utility teacher 来自 `rig_targets.candidate_utilities`。

## 5. CPC + GAIC 混训方式

配置：

```text
RIGCrop/configs/rig_crop_cpc_gaic_joint.yaml
```

支持：

```yaml
train_datasets:
  - jsonl_path: data/cpc_rig/metadata/train.jsonl
  - jsonl_path: data/gaic_rig/metadata/train.jsonl
```

训练脚本使用 `ConcatDataset` 混合样本，但每个样本的 supervision 来源不同：

```text
CPC: pairwise_preferences
GAIC: MOS-derived pairwise_preferences
Both: rig_targets graph supervision
```

## 6. 多卡训练

文件：

```text
RIGCrop/scripts/train_rig_crop.py
```

支持 `torchrun`：

```bash
torchrun --standalone --nproc_per_node=4 \
  RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_cpc_joint.yaml
```

实现细节：

- 自动读取 `WORLD_SIZE / RANK / LOCAL_RANK`
- NCCL 可用时使用 NCCL
- 使用 `DistributedSampler`
- 只有 rank0 写 checkpoint/history
- checkpoint 保存 unwrapped model 权重，方便单卡 eval/predict

## 7. 推理

文件：

```text
RIGCrop/scripts/predict_rig_crop.py
```

推理流程：

```text
image
  -> generate_anchors
  -> score anchors with RIGCropModel
  -> top-k boxes
```

命令：

```bash
python RIGCrop/scripts/predict_rig_crop.py \
  --image path/to/image.jpg \
  --checkpoint RIGCrop/runs/rig_crop_cpc_joint/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_joint.yaml \
  --out-json RIGCrop/runs/demo/pred.json \
  --out-vis RIGCrop/runs/demo/pred.jpg \
  --topk 5
```

这里不读取 Qwen JSONL，不读取 dataset candidates。

## 8. 当前实现边界

当前版本是可执行 first version：

- backbone 是轻量 TinyBackbone，方便 smoke/ablation。
- anchor generator 是内部 GAIC-style grid anchors。
- relation policy 来自规则化 Qwen middle state。

投稿前建议升级：

- backbone 换 ConvNeXt-T / Swin-T / DINOv2-S。
- 增加 Query-based crop generator。
- 加更强 graph visualization。
- 与 S2CNet / TransView / Cropper 做更完整对比。
