# RIGCrop 工作区

RIGCrop 是一个独立工作区，用于把当前项目已经生成的 DACC-style JSONL 和 Qwen/VLM 中间态转成可训练的结构化图监督，并训练一个推理时只输入图片的 image-only crop student。

核心约束：

```text
推理输入：一张图片
训练输入：现有 pipeline 输出的 JSONL + Qwen middle state + CPC/GAIC crop supervision
推理禁止：Qwen / VLM / SAM / detector / middle-state JSON
```

## 目录

```text
RIGCrop/
  configs/
    rig_crop_cpc_smoke.yaml
    rig_crop_cpc_joint.yaml
    rig_crop_cpc_gaic_joint.yaml
  rigcrop/
    schema.py          # DACC/Qwen -> rig_targets
    data.py            # CPC pairwise + GAIC MOS-derived pair dataset
    model.py           # RIGCropModel
    losses.py          # crop + graph + utility losses
    anchors.py         # image-only inference anchors
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

## 方法一句话

```text
RIG-Crop learns a latent entity-relation composition graph from VLM/Qwen
middle-state supervision during training, then uses this predicted graph to
score crop candidates at inference from image only.
```

## 最快 smoke test

在 repo 根目录执行：

```bash
bash RIGCrop/scripts/run_smoke_test.sh
```

## 服务器 4 卡训练

先准备 RIG targets：

```bash
bash RIGCrop/scripts/prepare_rig_targets.sh \
  data/cpc_semantic_qwen/metadata \
  data/cpc_rig/metadata
```

4 张 3090 后台训练：

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_joint.yaml
```

`configs/*.yaml` 里的 `batch_size` 是每个进程/每张 GPU 的 batch size。4 卡训练时全局 batch size 约为 `batch_size * 4`。

看日志：

```bash
tail -f RIGCrop/logs/rig_crop_train_*.log
```

## 关键原则

- CPC 和 GAIC 可以混训，但不能把 CPC pseudo score 与 GAIC MOS 直接合成一个普通回归标签。
- CPC 使用 `pairwise_preferences`。
- GAIC 使用 MOS 或 MOS-derived pairwise ranking。
- Qwen 中间态只作为 `rig_targets` 的 teacher supervision。
- `rig_targets` 是派生字段，不会修改原始 JSONL schema。

## 消融实验配置

```text
RIGCrop/configs/ablations/rig_crop_cpc_crop_only.yaml
RIGCrop/configs/ablations/rig_crop_cpc_no_importance.yaml
RIGCrop/configs/ablations/rig_crop_cpc_no_relation.yaml
RIGCrop/configs/ablations/rig_crop_cpc_no_utility.yaml
```

示例：

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/ablations/rig_crop_cpc_no_relation.yaml
```
