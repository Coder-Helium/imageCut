# RIGCrop / RIGFormer 执行文档

所有命令默认在 repo 根目录执行：

```bash
cd ~/workspace/imageCut
```

生产主模型已经重构为 RIGFormer。服务器建议安装：

```bash
pip install -r requirements-rigformer.txt
```

如果 DINOv3 权重已提前下载，把 `RIGCrop/configs/*.yaml` 里的：

```yaml
model:
  backbone:
    name: facebook/dinov3-vitb16-pretrain-lvd1689m
```

改为服务器本地权重目录：

```yaml
model:
  backbone:
    name: /path/to/local/dinov3-vitb16
```

如果你下载的是官方单文件 `.pth`，不要用 `dinov3_hf`，使用
`torchhub_dinov3`。需要本地 DINOv3 官方代码仓库和权重文件：

```yaml
model:
  backbone:
    type: torchhub_dinov3
    repo: /home/mx/dinov3
    source: local
    name: dinov3_vitb16
    pretrained: true
    weights: /home/mx/DINOv3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## 1. Smoke Test

用于验证代码可执行，不依赖真实数据：

一键：

```bash
bash RIGCrop/scripts/run_smoke_test.sh
```

分步：

```bash
python RIGCrop/scripts/create_smoke_data.py \
  --out-dir RIGCrop/runs/smoke_data

python RIGCrop/scripts/build_middle_state_targets.py \
  --input-jsonl RIGCrop/runs/smoke_data/train_qwen.jsonl \
  --out-jsonl RIGCrop/runs/smoke_data/train_rig.jsonl \
  --max-nodes 8 \
  --overwrite

python RIGCrop/scripts/build_middle_state_targets.py \
  --input-jsonl RIGCrop/runs/smoke_data/val_qwen.jsonl \
  --out-jsonl RIGCrop/runs/smoke_data/val_rig.jsonl \
  --max-nodes 8 \
  --overwrite

python RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_cpc_smoke.yaml

python RIGCrop/scripts/eval_rig_crop.py \
  --jsonl RIGCrop/runs/smoke_data/val_rig.jsonl \
  --checkpoint RIGCrop/runs/smoke_rig_crop/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_smoke.yaml \
  --batch-size 2 \
  --image-size 128 \
  --crop-size 96
```

推理 smoke：

```bash
python RIGCrop/scripts/predict_rig_crop.py \
  --image RIGCrop/runs/smoke_data/images/val_000.jpg \
  --checkpoint RIGCrop/runs/smoke_rig_crop/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_smoke.yaml \
  --out-json RIGCrop/runs/smoke_rig_crop/predict_val_000.json \
  --out-vis RIGCrop/runs/smoke_rig_crop/predict_val_000.jpg \
  --topk 3 \
  --image-size 128 \
  --crop-size 96
```

## 2. 准备 CPC Qwen 数据

如果还没跑 Qwen 版本，先跑：

```bash
mkdir -p logs data/cpc_semantic_qwen/metadata data/cpc_semantic_qwen/visualizations

nohup bash -lc 'set -euo pipefail; for split in train val; do python -u scripts/enrich_dacc_with_vlm_semantics.py --input-jsonl data/cpc_dacc/metadata/${split}.jsonl --out-jsonl data/cpc_semantic_qwen/metadata/${split}.jsonl --vlm qwen --qwen-model qwen3-vl-plus --resume --visualize --vis-dir data/cpc_semantic_qwen/visualizations/${split} --vis-topk 5 --sleep-sec 0.2; done' \
  > logs/cpc_qwen_semantic_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

看日志：

```bash
LATEST=$(ls -t logs/cpc_qwen_semantic_*.log | head -1)
tail -f "$LATEST"
```

## 3. Audit Qwen Schema

```bash
python RIGCrop/scripts/audit_middle_state_schema.py \
  --jsonl data/cpc_semantic_qwen/metadata/train.jsonl \
  --out-json data/cpc_rig/metadata/train.audit.json

python RIGCrop/scripts/audit_middle_state_schema.py \
  --jsonl data/cpc_semantic_qwen/metadata/val.jsonl \
  --out-json data/cpc_rig/metadata/val.audit.json
```

查看：

```bash
cat data/cpc_rig/metadata/train.audit.json
```

重点：

```text
main_subject_bbox
key_objects_bbox
top_relation_to_subject
top_suggested_actions
```

## 4. 构建 RIG Targets

一键：

```bash
bash RIGCrop/scripts/prepare_rig_targets.sh \
  data/cpc_semantic_qwen/metadata \
  data/cpc_rig/metadata
```

或手动：

```bash
python RIGCrop/scripts/build_middle_state_targets.py \
  --input-jsonl data/cpc_semantic_qwen/metadata/train.jsonl \
  --out-jsonl data/cpc_rig/metadata/train.jsonl \
  --max-nodes 12 \
  --progress-interval 200 \
  --overwrite

python RIGCrop/scripts/build_middle_state_targets.py \
  --input-jsonl data/cpc_semantic_qwen/metadata/val.jsonl \
  --out-jsonl data/cpc_rig/metadata/val.jsonl \
  --max-nodes 12 \
  --progress-interval 200 \
  --overwrite
```

检查：

```bash
wc -l data/cpc_rig/metadata/train.jsonl data/cpc_rig/metadata/val.jsonl
```

## 5. 单卡训练

```bash
python RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_cpc_joint.yaml
```

## 6. 4 张 3090 后台训练

推荐：

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_joint.yaml
```

说明：

```text
batch_size 是每张 GPU 的 batch size。
4 张 3090 时全局 batch size 约为 batch_size * 4。
如果显存富余，可以把 batch_size 从 16 提到 24/32。
```

等价命令：

```bash
mkdir -p RIGCrop/logs
nohup bash -lc 'set -euo pipefail; torchrun --standalone --nproc_per_node=4 RIGCrop/scripts/train_rig_crop.py --config RIGCrop/configs/rig_crop_cpc_joint.yaml' \
  > RIGCrop/logs/rig_crop_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

查看：

```bash
LATEST=$(ls -t RIGCrop/logs/rig_crop_train_*.log | head -1)
tail -f "$LATEST"
```

确认进程：

```bash
pgrep -af 'train_rig_crop|torchrun'
```

## 6.1 消融实验

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/ablations/rig_crop_cpc_crop_only.yaml

bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/ablations/rig_crop_cpc_no_importance.yaml

bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/ablations/rig_crop_cpc_no_relation.yaml

bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/ablations/rig_crop_cpc_no_utility.yaml
```

对应论文表：

```text
crop_only: 不用中间态监督
no_importance: 去掉 importance loss
no_relation: 去掉 relation loss
no_utility: 去掉 crop-conditioned utility distillation
full: rig_crop_cpc_joint.yaml
```

## 7. Eval

```bash
python RIGCrop/scripts/eval_rig_crop.py \
  --jsonl data/cpc_rig/metadata/val.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_cpc_joint/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_joint.yaml \
  --batch-size 32
```

## 8. Image-only Predict

```bash
python RIGCrop/scripts/predict_rig_crop.py \
  --image /path/to/test.jpg \
  --checkpoint RIGCrop/runs/rig_crop_cpc_joint/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_joint.yaml \
  --out-json RIGCrop/runs/demo/test.json \
  --out-vis RIGCrop/runs/demo/test.jpg \
  --topk 5
```

## 9. CPC + GAIC 混训

先分别生成：

```text
data/cpc_rig/metadata/train.jsonl
data/gaic_rig/metadata/train.jsonl
```

然后：

```bash
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

注意：

- CPC 使用原始 pairwise preference。
- GAIC 使用 candidate MOS 派生 pairwise。
- 两者共享 RIG graph supervision。
- 不要把 CPC pseudo score 和 GAIC MOS 合成同一个回归标签。
