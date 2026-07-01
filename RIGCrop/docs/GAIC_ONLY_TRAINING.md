# GAIC-only 训练说明

本文档用于只使用 GAICD 数据训练 RIGCrop / RIGFormer。

## 为什么先做 GAIC-only

你现在的现象是：

```text
CPC pairwise acc 还可以
GAICD Acc5 / Acc10 偏低
实际视觉效果里主体完整性和坏图改善不错
```

如果怀疑 CPC 数据质量或监督目标会干扰 GAICD 指标，那么先做 GAIC-only 是合理的。

GAIC-only 的好处：

```text
1. 训练目标完全贴近 GAICD MOS 排序
2. 不再被 CPC pair 数量主导
3. 更容易判断 GAICD Acc5 / Acc10 能否被 listwise/top-k loss 拉起来
4. 更适合作为 GAICD 指标优化实验
```

代价：

```text
1. 训练图像更少，容易过拟合
2. CPC 泛化指标可能下降
3. 需要更关注 val_gaic_acc5/acc10，而不是 train loss
```

## 配置文件

新增配置：

```text
RIGCrop/configs/rig_crop_gaic_only_dinov3_pth.yaml
```

该配置只使用：

```text
GAICD train pairwise
GAICD train listwise
GAICD val pairwise
GAICD val listwise
```

不再使用 CPC：

```text
没有 cpc_rig train
没有 CPC pairwise 混合
没有 repeat: 8
```

## 推荐参数

当前默认：

```yaml
batch_size: 16
gaic_listwise_batch_size: 1
epochs: 20
lr: 0.00003
best_metric: val_gaic_acc10
```

GAIC-only 数据量较小，建议先不要用很大学习率。

loss 默认更偏 GAICD：

```yaml
loss:
  crop_pair: 1.0
  gaic_listwise: 0.7
  gaic_topk: 0.7
  gaic_listwise_temperature: 0.30
  node: 0.1
  relation: 0.06
  utility: 0.15
  query: 0.05
```

中间态 loss 没有降权。

## 训练前检查

```bash
cd /root/image_cut
conda activate myenv

python - <<'PY'
import yaml

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_gaic_only_dinov3_pth.yaml"))
print("output_dir:", cfg["output_dir"])
print("best_metric:", cfg["best_metric"])
print("batch_size:", cfg["batch_size"])
print("gaic_pair_train:", cfg["train_datasets"][0]["jsonl_path"])
print("gaic_list_train:", cfg["gaic_listwise_train_dataset"]["jsonl_path"])
print("gaic_list_val:", cfg["gaic_listwise_val_dataset"]["jsonl_path"])
print("val_dataset:", cfg["val_dataset"]["jsonl_path"])
print("dinov3_repo:", cfg["model"]["backbone"]["repo"])
print("dinov3_weights:", cfg["model"]["backbone"]["weights"])
PY
```

检查数据读取：

```bash
PYTHONPATH=$PWD/RIGCrop python - <<'PY'
import yaml
from torch.utils.data import DataLoader
from rigcrop.data import RIGCandidateListDataset, RIGPairwiseDataset, collate_candidate_lists

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_gaic_only_dinov3_pth.yaml"))

d = dict(cfg["train_datasets"][0])
d["max_records"] = 8
ds = RIGPairwiseDataset(**d)
print("gaic pairwise pairs:", len(ds))
b = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
print("pairwise image:", b["image"].shape)
print("winner_box_feat:", b["winner_box_feat"].shape)

gd = dict(cfg["gaic_listwise_train_dataset"])
gd["max_records"] = 8
gds = RIGCandidateListDataset(**gd)
gb = next(iter(DataLoader(gds, batch_size=1, num_workers=0, collate_fn=collate_candidate_lists)))
print("gaic listwise records:", len(gds))
print("candidate_box_feats:", gb["candidate_box_feats"].shape)
print("candidate_scores:", gb["candidate_scores"].shape)
PY
```

## 单卡 smoke

生成 smoke 配置：

```bash
python - <<'PY'
import yaml
from pathlib import Path

src = Path("RIGCrop/configs/rig_crop_gaic_only_dinov3_pth.yaml")
dst = Path("RIGCrop/configs/rig_crop_gaic_only_dinov3_pth.smoke.yaml")
cfg = yaml.safe_load(src.read_text())

cfg["output_dir"] = "RIGCrop/runs/smoke_gaic_only"
cfg["batch_size"] = 2
cfg["gaic_listwise_batch_size"] = 1
cfg["gaic_val_batch_size"] = 1
cfg["num_workers"] = 0
cfg["persistent_workers"] = False
cfg["epochs"] = 1
cfg["log_interval"] = 1
cfg["train_datasets"][0]["max_records"] = 8
cfg["train_datasets"][0]["max_pairs_per_record"] = 16
cfg["val_dataset"]["max_records"] = 8
cfg["val_dataset"]["max_pairs_per_record"] = 16
cfg["gaic_listwise_train_dataset"]["max_records"] = 8
cfg["gaic_listwise_val_dataset"]["max_records"] = 8

dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(dst)
PY
```

运行 smoke，不加载旧 RIGCrop checkpoint：

```bash
PYTHONPATH=$PWD/RIGCrop CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_gaic_only_dinov3_pth.smoke.yaml
```

看到下面字段说明新版训练分支启用成功：

```text
gaic_total
gaic_lw
gaic_topk
gaic_acc5
gaic_acc10
val_gaic_acc5
val_gaic_acc10
```

## 正式后台训练

推荐用 screen：

```bash
cd /root/image_cut
conda activate myenv
screen -S gaic_only
```

进入 screen 后执行：

```bash
cd /root/image_cut
conda activate myenv
mkdir -p RIGCrop/logs

LOG=RIGCrop/logs/gaic_only_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --nproc_per_node=6 \
  --master_addr=127.0.0.1 \
  --master_port=29541 \
  RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_gaic_only_dinov3_pth.yaml \
  2>&1 | tee "$LOG"
```

退出 screen 但不停止训练：

```text
Ctrl + A
然后按 D
```

重新进入：

```bash
screen -r gaic_only
```

## 训练时看什么

主看：

```text
val_gaic_acc5
val_gaic_acc10
val_gaic_srcc
val_gaic_pcc
```

同时看中间态：

```text
val_node_loss
val_relation_loss
val_utility_loss
```

如果出现：

```text
train_gaic_acc5 上升
val_gaic_acc5 不升或下降
```

说明 GAIC-only 过拟合，需要降低 lr 或减少 epochs。

## 训练后评估

GAICD：

```bash
PYTHONPATH=$PWD/RIGCrop CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/test_script/eval_gaicd_official_metrics.py \
  --jsonl /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/val.compact.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_gaic_only_dinov3_pth/best.pt \
  --config RIGCrop/configs/rig_crop_gaic_only_dinov3_pth.yaml \
  --batch-size 256 \
  --image-size 384 \
  --crop-size 224 \
  --out-json RIGCrop/runs/rig_crop_gaic_only_dinov3_pth/gaic_val_metrics.json
```

CPC 可以作为泛化测试，但不是 GAIC-only 的主目标：

```bash
PYTHONPATH=$PWD/RIGCrop CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/test_script/eval_cpc_pairwise_metrics.py \
  --jsonl /root/autodl-tmp/data/beauty_dataset/data/cpc_rig/metadata/val.compact.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_gaic_only_dinov3_pth/best.pt \
  --config RIGCrop/configs/rig_crop_gaic_only_dinov3_pth.yaml \
  --batch-size 256 \
  --image-size 384 \
  --crop-size 224 \
  --out-json RIGCrop/runs/rig_crop_gaic_only_dinov3_pth/cpc_val_metrics.json
```

