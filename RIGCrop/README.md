# RIGCrop / RIGFormer 工作区

RIGCrop 现在默认实现为 **RIGFormer**：一个面向 AAAI 方法论文叙事的 VLM-distilled relation-importance graph transformer。它把当前项目已经生成的 DACC-style JSONL 和 Qwen/VLM 中间态转成结构化图监督，并训练一个推理时只输入图片的 image-only crop model。

核心约束：

```text
推理输入：一张图片
训练输入：现有 pipeline 输出的 JSONL + Qwen/VLM middle state + CPC/GAIC crop supervision
推理禁止：Qwen / VLM / SAM / detector / middle-state JSON / dataset candidates
```

## 方法一句话

```text
RIGFormer distills VLM/Qwen composition middle states into an image-only
entity-relation graph transformer, then uses graph-aware crop attention and
relation-preservation utility to score dense anchors and learned crop queries.
```

公开类名仍为 `RIGCropModel`，以兼容现有训练和推理脚本；内部架构已经替换为 RIGFormer。

## 架构

```text
image
  -> configurable foundation backbone
       compact_vit fallback / DINOv3 HF / DINOv3 timm / DINOv3 torchhub / ConvNeXt
  -> entity-relation Transformer decoder
       node boxes, roles, importance, validity
       typed relation policies and weights
  -> hybrid crop proposals
       dense anchors + learned crop query boxes
  -> graph-aware crop decision transformer
       crop token cross-attends predicted entity nodes
  -> learned relation-preservation utility + crop score
```

## 目录

```text
RIGCrop/
  configs/
    rig_crop_cpc_smoke.yaml
    rig_crop_cpc_joint.yaml
    rig_crop_gaic_joint.yaml
    rig_crop_cpc_gaic_joint.yaml
    ablations/
  rigcrop/
    backbones.py       # compact fallback + DINOv3/HF/timm/torchhub/ConvNeXt wrappers
    schema.py          # DACC/Qwen -> rig_targets
    data.py            # CPC pairwise + GAIC MOS-derived pair dataset
    model.py           # RIGCropModel public API, RIGFormer implementation
    losses.py          # crop + graph + utility losses
    anchors.py         # dense image-only inference anchors
  scripts/
    audit_middle_state_schema.py
    build_middle_state_targets.py
    prepare_rig_targets.sh
    run_smoke_test.sh
    train_rig_crop.py
    eval_rig_crop.py
    predict_rig_crop.py
    run_server_4gpu.sh
    create_smoke_data.py
  docs/
    DATA_PIPELINE_CONTRACT.md
    TECHNICAL_DESIGN.md
    EXECUTION.md
```

## DINOv3 / 服务器权重

主配置默认使用 Hugging Face 风格的 DINOv3：

```yaml
model:
  d_model: 384
  backbone:
    type: dinov3_hf
    name: facebook/dinov3-vitb16-pretrain-lvd1689m
    pretrained: true
    trust_remote_code: true
```

如果服务器已经下载好权重，把 `name` 改成本地目录即可：

```yaml
model:
  backbone:
    type: dinov3_hf
    name: /path/to/local/dinov3-vitb16
    pretrained: true
```

也支持 timm 或 torch.hub：

```yaml
backbone:
  type: dinov3_timm
  name: vit_base_patch16_dinov3.lvd1689m
  pretrained: true
```

```yaml
backbone:
  type: torchhub_dinov3
  repo: facebookresearch/dinov3
  name: dinov3_vitb16
  pretrained: true
```

如果你下载的是官方单文件 `.pth` 权重，使用 local torch.hub 配置。先把
`facebookresearch/dinov3` clone 到服务器，例如 `/home/mx/dinov3`，再配置：

```yaml
backbone:
  type: torchhub_dinov3
  repo: /home/mx/dinov3
  source: local
  name: dinov3_vitb16
  pretrained: true
  weights: /home/mx/DINOv3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

混合训练可直接使用：

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

本地 smoke test 使用 `compact_vit` fallback，不需要下载权重。

## 最快 smoke test

在 repo 根目录执行：

```bash
bash RIGCrop/scripts/run_smoke_test.sh
```

## 服务器 4 卡训练

先准备 RIG targets。生产默认建议 `MAX_NODES=12`：

```bash
MAX_NODES=12 \
bash RIGCrop/scripts/prepare_rig_targets.sh \
  data/cpc_semantic_qwen/metadata \
  data/cpc_rig/metadata
```

4 张 3090 后台训练：

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_joint.yaml
```

CPC + GAIC 混训：

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_joint.yaml
```

`configs/*.yaml` 里的 `batch_size` 是每个进程/每张 GPU 的 batch size。4 卡训练时全局 batch size 约为 `batch_size * 4`。

## 关键原则

- CPC 和 GAIC 可以混训，但不能把 CPC pseudo score 与 GAIC MOS 直接合成一个普通回归标签。
- CPC 使用 `pairwise_preferences`。
- GAIC 使用 MOS 或 MOS-derived pairwise ranking。
- Qwen/VLM 中间态只作为 `rig_targets` 的 teacher supervision。
- 推理时不读取 Qwen/VLM、中间态 JSON、detector、SAM 或 dataset candidates。
- Learned crop queries 只来自图像 token；推理脚本会把 query boxes 与 dense anchors 合并后统一打分。

## 消融实验配置

```text
RIGCrop/configs/ablations/rig_crop_cpc_crop_only.yaml
RIGCrop/configs/ablations/rig_crop_cpc_no_importance.yaml
RIGCrop/configs/ablations/rig_crop_cpc_no_relation.yaml
RIGCrop/configs/ablations/rig_crop_cpc_no_utility.yaml
```

这些消融共享同一个 RIGFormer/DINOv3 架构，只通过 loss 权重关闭 relation、importance、utility 等监督，适合直接形成论文消融表。
