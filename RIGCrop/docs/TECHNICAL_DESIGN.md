# RIGFormer 技术设计

日期：2026-06-23  
状态：RIGCrop 已重构为 RIGFormer，公开类名仍保留 `RIGCropModel` 以兼容现有脚本。

---

## 1. 目标

训练一个 **image-only graph transformer cropper**：

```text
input: image
output: top-k crop boxes + scores + relation-preservation utility
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
VLM
caption
middle-state JSON
detector
SAM
dataset candidates
```

---

## 2. 方法定位

RIGFormer 的论文定位：

```text
VLM-distilled Relation-Importance Graph Transformer for Image-Only Cropping
```

核心不是“用 VLM 生成伪标签”，而是：

```text
VLM/Qwen middle state
  -> entity roles, importance, bbox
  -> typed relation policy
  -> action policy
  -> crop-conditioned utility
  -> image-only transformer student
```

模型在推理时从图像 token 中预测 latent composition graph，并用该图结构指导 crop scoring。

---

## 3. 代码结构

```text
RIGCrop/rigcrop/
  backbones.py        # compact fallback + DINOv3/HF/timm/torchhub/ConvNeXt wrappers
  model.py            # RIGCropModel public API, RIGFormer implementation
  schema.py           # DACC/Qwen middle state -> rig_targets
  data.py             # RIGPairwiseDataset
  losses.py           # crop/graph/relation/utility/action losses
  anchors.py          # dense anchor generation
```

核心脚本：

```text
RIGCrop/scripts/build_middle_state_targets.py
RIGCrop/scripts/train_rig_crop.py
RIGCrop/scripts/eval_rig_crop.py
RIGCrop/scripts/predict_rig_crop.py
```

---

## 4. Backbone

文件：

```text
RIGCrop/rigcrop/backbones.py
```

支持后端：

| type | 用途 |
|---|---|
| `compact_vit` | 本地 smoke fallback，无需下载权重 |
| `dinov3_hf` | Hugging Face / local path DINOv3 |
| `dinov3_timm` | timm DINOv3 |
| `torchhub_dinov3` | Meta torch.hub DINOv3, including local `.pth` weights |
| `torchvision_convnext` | torchvision ConvNeXt fallback |

主配置示例：

```yaml
model:
  d_model: 384
  backbone:
    type: dinov3_hf
    name: /path/to/local/dinov3-vitb16
    pretrained: true
    trust_remote_code: true
```

官方 `.pth` 单文件权重示例：

```yaml
model:
  d_model: 384
  backbone:
    type: torchhub_dinov3
    repo: /home/mx/dinov3
    source: local
    name: dinov3_vitb16
    pretrained: true
    weights: /home/mx/DINOv3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

`backbone` 输出统一为：

```python
BackboneOutput(
    tokens: B x N x D,
    pooled: B x D,
    spatial_size: (Hf, Wf),
)
```

---

## 5. Entity-Relation Transformer

文件：

```text
RIGCrop/rigcrop/model.py
```

入口：

```python
RIGCropModel.encode_graph(image)
```

流程：

```text
image
  -> backbone tokens
  -> learnable entity queries
  -> TransformerDecoder(entity queries, image tokens)
  -> node heads
  -> relation heads
```

输出节点：

```text
node_boxes        B x M x 4
node_role_logits  B x M x |ROLES|
node_importance   B x M
node_valid_logits B x M
```

输出关系：

```text
relation_logits B x M x M x |RELATION_POLICIES|
relation_weight B x M x M
```

relation head 输入包含：

```text
node_i
node_j
node_i * node_j
|node_i - node_j|
relative geometry embedding
```

这取代了旧版：

```text
global pooled vector -> MLP -> node tokens
```

---

## 6. Graph-Aware Crop Decision Transformer

训练和推理仍然调用：

```python
model(image, crop, box_feat, graph=None)
```

在 `crop_feature_mode: roi_tokens` 下，`crop` 可以为 `None`；crop 表征来自 full-image token map 的 RoI pooling。旧版独立 crop backbone 可通过 `crop_feature_mode: crop_backbone` 恢复。

内部流程：

```text
full image
  -> backbone tokens
  -> token-space RoI pooling(candidate box)
  -> crop pooled token
  + box geometry embedding
  + relation-preservation utility embedding
  -> crop token
  -> cross-attend predicted node tokens
  -> crop score + utility
```

当前 graph-aware scorer 使用：

```text
_CropGraphAttentionBlock
  MultiheadAttention(crop_token -> node_tokens)
  FFN
  relation-preservation gate from graph_feat
```

输出：

```text
score   B
utility B
graph_feat B x 5
crop_state B x D
```

---

## 7. Learned Crop Queries

RIGFormer 同时预测 learned crop query boxes：

```text
crop_queries
  -> TransformerDecoder(crop queries, image tokens)
  -> query_boxes
  -> query_scores
```

在推理脚本中：

```text
dense anchors
  + query_boxes
  -> merge/dedupe
  -> graph-aware score
  -> top-k
```

对应文件：

```text
RIGCrop/scripts/predict_rig_crop.py
```

这样主方法不再是纯固定 anchor ranker，而是 hybrid anchor-query cropper。

---

## 8. Relation-Preservation Utility

`graph_features_for_crop` 保留可解释的 5 个分量：

```text
node_keep
relation_keep
distractor
boundary
main_cov
```

这些分量来自 predicted graph 与 candidate crop 的几何关系：

```text
coverage(node_box, crop_box)
relation_preserve_prob
relation_weight
distractor_prob
boundary_cut
main_subject_prob
```

与旧版不同：

- 旧版直接把 5 维 feature 交给 MLP score；
- 新版先把 5 维 feature 投影成 utility embedding；
- crop token cross-attends graph tokens；
- 5 维 feature 同时作为 relation-preservation gate 和 score/utility head 输入。

---

## 9. Loss

文件：

```text
RIGCrop/rigcrop/losses.py
RIGCrop/scripts/train_rig_crop.py
```

当前训练总目标：

```text
L =
  crop_pair * L_crop_pair
  + node_bbox/node_role/node_importance/node_valid
  + relation_policy/relation_weight
  + utility
  + action
```

训练脚本现在会对每个 batch：

```python
graph = model.encode_graph(image)
winner = model(None, None, winner_box_feat, graph=graph)
loser  = model(None, None, loser_box_feat, graph=graph)
```

避免 winner/loser 额外编码 crop image，使 DINOv3 从约 3 次 backbone / sample 降为 1 次 backbone / sample。详见：

```text
RIGCrop/docs/TOKEN_ROI_CROP_POOLING.md
```

---

## 10. RIG Targets

生产默认：

```text
max_nodes = 12
```

文件：

```text
RIGCrop/scripts/build_middle_state_targets.py
RIGCrop/scripts/prepare_rig_targets.sh
```

smoke test 显式使用 `--max-nodes 8`，保持本地快速验证。

---

## 11. 配置

### Smoke

```text
RIGCrop/configs/rig_crop_cpc_smoke.yaml
```

使用：

```yaml
backbone:
  type: compact_vit
```

### 主实验

```text
RIGCrop/configs/rig_crop_cpc_joint.yaml
RIGCrop/configs/rig_crop_gaic_joint.yaml
RIGCrop/configs/rig_crop_cpc_gaic_joint.yaml
```

默认：

```yaml
model:
  d_model: 384
  max_nodes: 12
  backbone:
    type: dinov3_hf
    name: facebook/dinov3-vitb16-pretrain-lvd1689m
    pretrained: true
  num_entity_layers: 6
  num_crop_layers: 3
  num_crop_queries: 16
```

服务器本地权重：

```yaml
name: /path/to/local/dinov3-vitb16
```

---

## 12. 推理

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

推理流程：

```text
image
  -> encode graph once
  -> generate dense anchors
  -> append learned query boxes
  -> score candidates with graph-aware crop transformer
  -> sort by score
```

---

## 13. 与旧版 RIGCrop 的差异

| 旧版 | 新版 RIGFormer |
|---|---|
| Tiny CNN | configurable foundation backbone, DINOv3-ready |
| pooled feature -> MLP nodes | learnable entity queries + Transformer decoder |
| pairwise MLP relations | relation head with token and geometry features |
| anchors only | dense anchors + learned crop query boxes |
| concat score head | crop token cross-attends graph nodes |
| fixed 8 nodes | production default 12 nodes |
| graph encoded twice per pair | graph encoded once and reused |

---

## 14. AAAI 叙事

推荐方法表述：

```text
RIGFormer is not a VLM cropper at inference.
It distills VLM-generated explicit composition reasoning into an image-only
entity-relation graph transformer, and uses graph-aware crop attention plus
relation-preservation utility to make crop decisions.
```

核心贡献：

1. VLM middle-state graph distillation；
2. Entity-relation Transformer graph prediction；
3. Graph-aware crop decision transformer；
4. Hybrid anchor-query image-only cropping；
5. Relation-preservation utility for interpretability and supervision。
