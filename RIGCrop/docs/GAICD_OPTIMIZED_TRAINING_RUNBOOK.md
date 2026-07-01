# GAICD 指标优化训练手册

本文档用于新的 GAICD 优化训练流程。当前训练代码已经从原来的
“CPC+GAICD 混合 pairwise 训练”升级为：

```text
中间态监督保持不降权
+ CPC/GAICD pairwise ranking
+ GAICD listwise ranking
+ GAICD top-k hard negative ranking
+ 按 GAICD Acc10 保存 best
```

目标是让模型更贴近 GAICD 的论文指标：

```text
Acc5 / Acc10 / SRCC / PCC
```

同时保留你的论文核心：**中间态引导的语义安全裁剪**。

## 0. 默认服务器环境

本文档默认你的服务器路径如下：

```text
项目路径:        /root/workspace/image_cut
conda 环境:      myenv
数据根目录:      /root/autodl-tmp/data/beauty_dataset
DINOv3 权重:     /root/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
DINOv3 repo:     /root/dinov3
GPU:             6 x RTX 4090, 24GB
```

如果你的路径不同，只需要把下面命令里的路径替换成你的实际路径。

## 1. 同步最新代码

服务器上执行：

```bash
cd /root/workspace/image_cut
git pull
```

如果服务器不是 git 管理的代码，需要至少同步这些文件：

```text
RIGCrop/rigcrop/data.py
RIGCrop/rigcrop/losses.py
RIGCrop/scripts/train_rig_crop.py
RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
RIGCrop/docs/GAICD_OPTIMIZED_TRAINING.md
RIGCrop/docs/GAICD_OPTIMIZED_TRAINING_RUNBOOK.md
```

## 2. 修改主配置文件

打开配置：

```bash
cd /root/workspace/image_cut
vim RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

重点确认这些字段。

### 2.1 输出目录

建议不要覆盖旧训练目录，换一个新目录：

```yaml
output_dir: RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt
best_metric: val_gaic_acc10
```

含义：

```text
best.pt      按 val_gaic_acc10 保存
best_gaic.pt 同样偏 GAICD 指标
last.pt      最后一轮
```

### 2.2 数据路径

AutoDL 上建议使用 compact jsonl：

```yaml
train_datasets:
  - jsonl_path: /root/autodl-tmp/data/beauty_dataset/data/cpc_rig/metadata/train.compact.jsonl
    image_size: 384
    crop_size: 224
    max_records:
    max_pairs_per_record: 128
    max_nodes: 12
    derive_pairs_from_scores: true
    image_cache_size: 8
    compact_records: true
    return_crops: false

  - jsonl_path: /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/train.compact.jsonl
    repeat: 8
    image_size: 384
    crop_size: 224
    max_records:
    max_pairs_per_record: 128
    max_nodes: 12
    derive_pairs_from_scores: true
    min_score_gap: 0.05
    image_cache_size: 8
    compact_records: true
    return_crops: false
```

这里的 `repeat: 8` 是为了提高 GAICD pairwise 分支的出现频率。原来的
CPC:GAICD pair 数量大约是 9:1，GAICD 对训练影响太弱。

### 2.3 GAICD listwise 数据

新增的 GAICD listwise 分支需要单独配置：

```yaml
gaic_listwise_train_dataset:
  jsonl_path: /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/train.compact.jsonl
  image_size: 384
  max_records:
  max_nodes: 12
  max_candidates:
  compact_records: true
  image_cache_size: 8

gaic_listwise_val_dataset:
  jsonl_path: /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/val.compact.jsonl
  image_size: 384
  max_records:
  max_nodes: 12
  max_candidates:
  compact_records: true
  image_cache_size: 8
```

这个分支每次读取一张图的所有 GAICD 候选框和 MOS 分数，用来优化：

```text
Acc5 / Acc10 / SRCC / PCC
```

### 2.4 CPC 验证集

当前训练中的 `val_dataset` 仍然是 CPC pairwise 验证：

```yaml
val_dataset:
  jsonl_path: /root/autodl-tmp/data/beauty_dataset/data/cpc_rig/metadata/val.compact.jsonl
  image_size: 384
  crop_size: 224
  max_records:
  max_pairs_per_record: 128
  max_nodes: 12
  derive_pairs_from_scores: true
  image_cache_size: 8
  compact_records: true
  return_crops: false
```

所以每个 epoch 会同时得到：

```text
val_pairwise_acc       CPC 风格 pairwise 指标
val_gaic_acc5          GAICD Acc5
val_gaic_acc10         GAICD Acc10
val_gaic_srcc          GAICD SRCC
val_gaic_pcc           GAICD PCC
```

### 2.5 DINOv3 路径

AutoDL 上建议：

```yaml
model:
  backbone:
    type: torchhub_dinov3
    repo: /root/dinov3
    source: local
    name: dinov3_vitb16
    pretrained: true
    weights: /root/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

### 2.6 推荐训练参数

因为新增了 GAICD listwise 分支，一次会给约 90 个候选框打分，显存会比之前更高。
6 卡 4090 建议先从保守设置开始：

```yaml
batch_size: 16
gaic_listwise_batch_size: 1
gaic_val_batch_size: 1
num_workers: 6
persistent_workers: true
prefetch_factor: 2
epochs: 10
log_interval: 500
lr: 0.00003
weight_decay: 0.0001
```

稳定后可以尝试：

```yaml
batch_size: 20
```

如果显存仍然稳定，再尝试：

```yaml
batch_size: 24
```

不建议一开始直接 `batch_size: 40`，因为新分支比之前更吃显存。

### 2.7 loss 设置

中间态 loss 不降权，保持你的论文核心：

```yaml
loss:
  crop_pair: 1.0
  gaic_listwise: 0.5
  gaic_topk: 0.5
  gaic_listwise_temperature: 0.35
  gaic_topk_positive: 5
  gaic_topk_negative_after: 10
  gaic_topk_margin: 1.0
  node: 0.1
  relation: 0.06
  utility: 0.15
  query: 0.05
  action: 0.0
```

含义：

```text
crop_pair       CPC/GAICD pairwise ranking
gaic_listwise   GAICD 整组候选框 MOS 排序
gaic_topk       专门优化 top5/top10 候选框检索
node            中间态节点 bbox/role/importance/valid
relation        中间态关系 policy/weight
utility         中间态 utility 蒸馏
query           crop query proposal
```

## 3. 训练前检查

进入环境：

```bash
cd /root/workspace/image_cut
conda activate myenv
```

### 3.1 检查文件是否存在

```bash
ls -lh /root/autodl-tmp/data/beauty_dataset/data/cpc_rig/metadata/train.compact.jsonl
ls -lh /root/autodl-tmp/data/beauty_dataset/data/cpc_rig/metadata/val.compact.jsonl
ls -lh /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/train.compact.jsonl
ls -lh /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/val.compact.jsonl
ls -lh /root/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

### 3.2 检查配置是否正确

```bash
python - <<'PY'
import yaml

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml"))
print("output_dir:", cfg["output_dir"])
print("best_metric:", cfg.get("best_metric"))
print("batch_size:", cfg["batch_size"])
print("gaic_listwise_batch_size:", cfg["gaic_listwise_batch_size"])
print("cpc_train:", cfg["train_datasets"][0]["jsonl_path"])
print("gaic_pair_train:", cfg["train_datasets"][1]["jsonl_path"])
print("gaic_list_train:", cfg["gaic_listwise_train_dataset"]["jsonl_path"])
print("gaic_list_val:", cfg["gaic_listwise_val_dataset"]["jsonl_path"])
print("cpc_val:", cfg["val_dataset"]["jsonl_path"])
print("dinov3_repo:", cfg["model"]["backbone"]["repo"])
print("dinov3_weights:", cfg["model"]["backbone"]["weights"])
PY
```

### 3.3 检查数据 pipeline

```bash
PYTHONPATH=$PWD/RIGCrop python - <<'PY'
import yaml
from torch.utils.data import DataLoader
from rigcrop.data import RIGCandidateListDataset, RIGPairwiseDataset, collate_candidate_lists

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml"))

for i, d in enumerate(cfg["train_datasets"]):
    d = dict(d)
    d.pop("repeat", None)
    d["max_records"] = 4
    ds = RIGPairwiseDataset(**d)
    print("pairwise train", i, "pairs", len(ds), "path", d["jsonl_path"])
    b = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
    print("  image", b["image"].shape)
    print("  winner_box_feat", b["winner_box_feat"].shape)
    print("  loser_box_feat", b["loser_box_feat"].shape)

gd = dict(cfg["gaic_listwise_train_dataset"])
gd["max_records"] = 4
gds = RIGCandidateListDataset(**gd)
gb = next(iter(DataLoader(gds, batch_size=1, num_workers=0, collate_fn=collate_candidate_lists)))
print("gaic listwise records", len(gds))
print("  image", gb["image"].shape)
print("  candidate_box_feats", gb["candidate_box_feats"].shape)
print("  candidate_scores", gb["candidate_scores"].shape)
print("  candidate_mask", gb["candidate_mask"].shape)
PY
```

正常情况下能看到：

```text
pairwise train 0 pairs ...
pairwise train 1 pairs ...
gaic listwise records ...
candidate_box_feats torch.Size([1, 90, 8])
```

不同 GAICD 版本候选框数量可能不是 90，没关系。

### 3.4 检查模型能否跑 GAICD listwise forward

```bash
PYTHONPATH=$PWD/RIGCrop CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import yaml
import torch
from torch.utils.data import DataLoader
from rigcrop.data import RIGCandidateListDataset, collate_candidate_lists
from rigcrop.model import RIGCropModel

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml"))
ds_cfg = dict(cfg["gaic_listwise_train_dataset"])
ds_cfg["max_records"] = 2
ds = RIGCandidateListDataset(**ds_cfg)
b = next(iter(DataLoader(ds, batch_size=1, num_workers=0, collate_fn=collate_candidate_lists)))

device = "cuda"
model = RIGCropModel(**cfg["model"]).to(device).eval()
image = b["image"].to(device)
box_feats = b["candidate_box_feats"].to(device)
B, C, D = box_feats.shape

with torch.no_grad():
    graph = model.encode_graph(image)
    flat_graph = {
        k: (v.repeat_interleave(C, 0) if torch.is_tensor(v) else v)
        for k, v in graph.items()
    }
    out = model(None, None, box_feats.reshape(B * C, D), graph=flat_graph)
    scores = out.get("score_logit", out["score"]).view(B, C)

print("OK scores", scores.shape)
print("mem_allocated_GB", torch.cuda.memory_allocated() / 1024**3)
print("mem_reserved_GB", torch.cuda.memory_reserved() / 1024**3)
PY
```

如果这里失败，先不要正式训练。

## 4. 单卡 smoke 训练

正式 6 卡前，先做一个很小的 smoke。

### 4.1 生成 smoke 配置

```bash
python - <<'PY'
import yaml
from pathlib import Path

src = Path("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml")
dst = Path("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.smoke.yaml")
cfg = yaml.safe_load(src.read_text())

cfg["output_dir"] = "RIGCrop/runs/smoke_gaic_opt"
cfg["batch_size"] = 2
cfg["gaic_listwise_batch_size"] = 1
cfg["gaic_val_batch_size"] = 1
cfg["num_workers"] = 0
cfg["persistent_workers"] = False
cfg["prefetch_factor"] = 2
cfg["epochs"] = 1
cfg["log_interval"] = 1

for d in cfg["train_datasets"]:
    d["max_records"] = 8
    d["max_pairs_per_record"] = 16

cfg["val_dataset"]["max_records"] = 8
cfg["val_dataset"]["max_pairs_per_record"] = 16
cfg["gaic_listwise_train_dataset"]["max_records"] = 8
cfg["gaic_listwise_val_dataset"]["max_records"] = 8

dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(dst)
PY
```

### 4.2 跑 smoke

如果你要从旧 checkpoint 继续：

```bash
PYTHONPATH=$PWD/RIGCrop CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.smoke.yaml \
  --resume RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt
```

如果旧 checkpoint 不在这个路径，改成你的实际路径。

成功时日志应该出现：

```text
gaic_total
gaic_lw
gaic_topk
gaic_acc5
gaic_acc10
```

epoch 结束后应该出现：

```text
val_gaic_acc5
val_gaic_acc10
val_gaic_srcc
val_gaic_pcc
```

## 5. 正式 6 卡训练

推荐从旧的 `best.pt` 继续 fine-tune，但输出到新目录：

```yaml
output_dir: RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt
```

启动：

```bash
cd /root/workspace/image_cut
conda activate myenv
mkdir -p RIGCrop/logs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --nproc_per_node=6 \
  --master_addr=127.0.0.1 \
  --master_port=29531 \
  RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --resume RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt \
  > RIGCrop/logs/gaic_opt_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

查看日志：

```bash
tail -f $(ls -t RIGCrop/logs/gaic_opt_*.log | head -1)
```

如果你不需要 resume，就去掉最后一行的 `--resume ...`。

## 6. 训练过程中怎么看

### 6.1 GPU 状态

```bash
nvitop
```

重点看：

```text
GPU-Util       尽量 90%+
Memory-Usage   4090 不要长期贴近 23.8GB+
温度           4090 长期 85 度以上要注意
```

### 6.2 训练进程

```bash
ps aux | grep train_rig_crop
```

### 6.3 最近几轮指标

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt/history.json")
if not p.exists():
    print("history not found:", p)
    raise SystemExit

h = json.load(open(p))
for x in h[-5:]:
    print({
        "epoch": x.get("epoch"),
        "val_gaic_acc5": x.get("val_gaic_acc5"),
        "val_gaic_acc10": x.get("val_gaic_acc10"),
        "val_gaic_srcc": x.get("val_gaic_srcc"),
        "val_gaic_pcc": x.get("val_gaic_pcc"),
        "val_pairwise_acc": x.get("val_pairwise_acc"),
        "val_node_loss": x.get("val_node_loss"),
        "val_relation_loss": x.get("val_relation_loss"),
    })
PY
```

### 6.4 训练曲线

训练会自动生成：

```text
RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt/training_curves.png
RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt/gaic_acc5.png
RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt/gaic_acc10.png
RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt/pairwise_acc.png
```

## 7. 应该看哪些指标

GAICD 主指标：

```text
val_gaic_acc5      越高越好
val_gaic_acc10     越高越好，当前 best.pt 默认按它保存
val_gaic_srcc      越高越好
val_gaic_pcc       越高越好
```

CPC 辅助指标：

```text
val_pairwise_acc   不要大幅下降
val_score_margin   不要明显变负
```

中间态稳定性：

```text
val_node_loss
val_relation_loss
val_utility_loss
```

中间态是你的论文核心，所以不能只看 GAICD 指标。如果 GAICD 上升但
`val_node_loss / val_relation_loss` 爆炸，也要谨慎。

## 8. 什么时候算有效提升

你之前大概是：

```text
GAICD Acc5  ≈ 0.288
GAICD Acc10 ≈ 0.466
GAICD SRCC  ≈ 0.710
CPC Acc     ≈ 0.761
```

比较理想的早期信号：

```text
val_gaic_acc10 > 0.50
val_gaic_acc5  > 0.32
val_gaic_srcc  维持在 0.70 左右或更高
val_pairwise_acc 不低于 0.72
val_node_loss / val_relation_loss 不爆炸
```

如果 3 个 epoch 后：

```text
val_gaic_acc10 没涨
val_gaic_acc5 没涨
train_gaic_acc5 涨但 val_gaic_acc5 不涨
```

说明可能过拟合 GAICD train，需要降低学习率或降低 `gaic_topk`。

## 9. checkpoint 怎么选

训练目录里会有：

```text
best.pt
best_gaic.pt
best_pairwise.pt
last.pt
history.json
training_curves.png
```

用途：

```text
best.pt          默认按 val_gaic_acc10 保存，用于 GAICD 主表
best_gaic.pt     GAICD 指标最好时保存，通常和 best.pt 接近
best_pairwise.pt CPC pairwise 最好，用于 CPC 表
last.pt          最新状态，不一定最好
```

投稿主结果建议：

```text
GAICD: 用 best.pt 或 best_gaic.pt
CPC:   用 best_pairwise.pt，同时报告 best.pt 的 CPC 指标作为补充
```

## 10. 训练后正式评估

### 10.1 GAICD 评估

```bash
mkdir -p RIGCrop/runs/gaicd_eval

PYTHONPATH=$PWD/RIGCrop CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/test_script/eval_gaicd_official_metrics.py \
  --jsonl /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/val.compact.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --batch-size 256 \
  --image-size 384 \
  --crop-size 224 \
  --out-json RIGCrop/runs/gaicd_eval/gaicd_val_metrics_gaic_opt.json
```

输出里重点看：

```text
Acc5
Acc10
SRCC
PCC
```

### 10.2 CPC 评估

```bash
mkdir -p RIGCrop/runs/cpc_eval

PYTHONPATH=$PWD/RIGCrop CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/test_script/eval_cpc_pairwise_metrics.py \
  --jsonl /root/autodl-tmp/data/beauty_dataset/data/cpc_rig/metadata/val.compact.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt/best_pairwise.pt \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --batch-size 256 \
  --image-size 384 \
  --crop-size 224 \
  --out-json RIGCrop/runs/cpc_eval/cpc_val_metrics_gaic_opt.json
```

输出里重点看：

```text
pairwise_acc
swap_error
weighted_pairwise_acc
```

## 11. 显存不够怎么办

如果 OOM，先改：

```yaml
batch_size: 12
num_workers: 4
prefetch_factor: 2
gaic_listwise_batch_size: 1
```

不要优先改 `gaic_listwise_batch_size` 到更大，它最容易增加显存。

如果显存很空，比如低于 20GB：

```yaml
batch_size: 20
```

稳定后再试：

```yaml
batch_size: 24
```

## 12. 指标不涨怎么办

### 情况 A：GAICD 不涨，CPC 还可以

提高 GAICD loss：

```yaml
loss:
  gaic_listwise: 0.7
  gaic_topk: 0.7
  gaic_listwise_temperature: 0.25
```

### 情况 B：GAICD 涨，CPC 掉太多

降低 GAICD 强度：

```yaml
loss:
  gaic_listwise: 0.3
  gaic_topk: 0.4
```

或者降低 GAICD pairwise repeat：

```yaml
repeat: 4
```

### 情况 C：中间态 loss 爆炸

不要立刻降中间态权重。先降低学习率：

```yaml
lr: 0.00001
```

如果还不稳，再考虑冻结 backbone 或缩小 GAICD top-k loss。

## 13. 如何停止训练

查看进程：

```bash
ps aux | grep train_rig_crop
```

停止训练：

```bash
pkill -f "RIGCrop/scripts/train_rig_crop.py"
pkill -f "torchrun.*train_rig_crop.py"
```

确认已经停止：

```bash
ps aux | grep train_rig_crop
```

只剩 `grep train_rig_crop` 就说明停干净了。

