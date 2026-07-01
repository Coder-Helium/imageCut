# RIGFormer Server Validation Pipeline

本文档维护服务器迁移和代码更新后的完整流程验证 pipeline，重点覆盖当前 DINOv3 + Token-RoI Crop Pooling 版本。

目标不是验证最终精度，而是在正式付费启动 GPU 长训前，确认以下链路全部可用：

```text
环境依赖
代码导入
JSONL 路径
CPC / GAIC compact 数据
Dataset 读取
Token-RoI crop scoring
DINOv3 权重加载
单卡训练冒烟
6 卡 DDP 启动
日志与资源监控
```

## 1. 默认路径约定

AutoDL / root 容器常用路径：

```bash
export PROJECT=/root/workspace/image_cut
export DATA=/root/autodl-tmp/data/beauty_dataset
export CONDA_ENV=myenv
export CONFIG=RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
export DINO_REPO=/root/dinov3
export DINO_WEIGHTS=/root/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

如果你的项目目录是 `/root/workspace/imageCut`，把 `PROJECT` 改成真实目录即可。

进入项目：

```bash
conda activate "$CONDA_ENV"
cd "$PROJECT"
```

## 2. 当前代码版本必须包含的关键配置

当前优化版应该使用 Token-RoI 单次 backbone 路径：

```yaml
model:
  crop_feature_mode: roi_tokens
  crop_pool_size: 4
```

训练数据建议关闭 crop tensor 生成：

```yaml
train_datasets:
  - return_crops: false
  - return_crops: false

val_dataset:
  return_crops: false
```

检查命令：

```bash
grep -nE "crop_feature_mode|crop_pool_size|return_crops|jsonl_path|repo:|weights:" "$CONFIG"
```

通过标准：

```text
crop_feature_mode: roi_tokens
crop_pool_size: 4
return_crops: false
jsonl_path 指向新服务器 DATA 路径
repo 指向 DINOv3 代码目录
weights 指向 DINOv3 .pth 权重
```

## 3. 无 GPU 前置验证

服务器还没开 GPU 时，先跑这一组。此阶段允许 `torch.cuda.is_available()` 为 `False`。

### 3.1 基础依赖导入

```bash
conda activate "$CONDA_ENV"
cd "$PROJECT"

PYTHONPATH=$PWD/RIGCrop python - <<'PY'
import sys
import yaml
import cv2
import numpy as np
import torch

print("python", sys.version)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("yaml ok")
print("cv2", cv2.__version__)
print("numpy", np.__version__)
PY
```

如果报：

```text
ModuleNotFoundError: No module named 'cv2'
```

安装：

```bash
python -m pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
```

不要执行：

```bash
pip install cv2
```

`cv2` 是导入名，不是 pip 包名。

### 3.2 Python 语法检查

```bash
python -m compileall -q RIGCrop/rigcrop RIGCrop/scripts
```

通过标准：无输出、返回 shell prompt。

### 3.3 配置旧路径扫描

```bash
grep -R "/home/hmx/nas\|/dmci_user/mx\|/home/mx\|/Users/hmx" \
  RIGCrop/configs 2>/dev/null || true
```

如果这是 AutoDL root 环境，主训练 config 中不应该再出现旧服务器数据路径。DINOv3 repo / weights 也必须是当前服务器路径。

## 4. CPC / GAIC JSONL 路径修复

换服务器后，JSONL 内部的 `image_path` 常常仍然是旧路径。先抽样看当前真实旧前缀。

```bash
python - <<'PY'
import json
from pathlib import Path

DATA = Path("/root/autodl-tmp/data/beauty_dataset")
files = [
    DATA / "data/cpc_rig/metadata/train.jsonl",
    DATA / "data/cpc_rig/metadata/val.jsonl",
    DATA / "data/gaic_rig/metadata/train.jsonl",
    DATA / "data/gaic_rig/metadata/val.jsonl",
]

for p in files:
    print("\n==", p)
    if not p.exists():
        print("missing jsonl")
        continue
    with p.open("r", encoding="utf-8") as f:
        for _, line in zip(range(3), f):
            r = json.loads(line)
            print(r.get("sample_id"), r.get("image_path"))
PY
```

常见旧路径有两类：

```text
/home/hmx/nas/beauty_dataset
/dmci_user/mx/ocean_nas/beauty_dataset
```

AutoDL 新路径通常是：

```text
/root/autodl-tmp/data/beauty_dataset
```

### 4.1 流式修复 CPC + GAIC

如果原始 `rewrite_jsonl_image_paths.py` 在大 train JSONL 上被 `Killed`，用下面这个流式版本，避免把所有 records 一次性读进内存。

根据抽样结果设置 `OLD`。如果抽样显示旧路径是 `/home/hmx/nas/beauty_dataset`：

```bash
export DATA=/root/autodl-tmp/data/beauty_dataset
export OLD=/home/hmx/nas/beauty_dataset
export NEW=/root/autodl-tmp/data/beauty_dataset
```

如果抽样显示旧路径是 `/dmci_user/mx/ocean_nas/beauty_dataset`：

```bash
export DATA=/root/autodl-tmp/data/beauty_dataset
export OLD=/dmci_user/mx/ocean_nas/beauty_dataset
export NEW=/root/autodl-tmp/data/beauty_dataset
```

执行流式修复：

```bash
python - <<'PY'
import json, os
from pathlib import Path

DATA = Path(os.environ["DATA"])
OLD = os.environ["OLD"].rstrip("/")
NEW = os.environ["NEW"].rstrip("/")

files = [
    DATA / "data/cpc_rig/metadata/train.jsonl",
    DATA / "data/cpc_rig/metadata/val.jsonl",
    DATA / "data/gaic_rig/metadata/train.jsonl",
    DATA / "data/gaic_rig/metadata/val.jsonl",
]

for inp in files:
    if not inp.exists():
        print(f"[skip] {inp} not found")
        continue

    out = inp.with_name(inp.stem + ".fixed.jsonl")
    tmp = out.with_suffix(out.suffix + ".tmp")

    processed = changed = missing = 0
    examples = []
    missing_examples = []

    with inp.open("r", encoding="utf-8") as f, tmp.open("w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            old_path = str(r.get("image_path", ""))
            new_path = old_path

            if old_path.startswith(OLD):
                new_path = NEW + old_path[len(OLD):]

            r["image_path"] = new_path
            processed += 1

            if new_path != old_path:
                changed += 1
                if len(examples) < 3:
                    examples.append((old_path, new_path))

            if not Path(new_path).exists():
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(new_path)

            w.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    tmp.replace(out)

    print("\n==", inp)
    print("out:", out)
    print("processed:", processed, "changed:", changed, "missing:", missing)
    print("examples:", examples)
    print("missing_examples:", missing_examples)
PY
```

通过标准：

```text
missing: 0
changed: 大于 0
```

如果 `changed: 0` 且 `missing` 很多，说明 `OLD` 前缀写错了，回到抽样命令重新确认。

### 4.2 替换原 JSONL

只有在 `missing=0` 后再替换。

```bash
for ds in cpc_rig gaic_rig; do
  for split in train val; do
    src="$DATA/data/$ds/metadata/$split.jsonl"
    fixed="$DATA/data/$ds/metadata/$split.fixed.jsonl"

    [ -f "$fixed" ] || continue

    cp "$src" "$src.bak_$(date +%Y%m%d_%H%M%S)"
    mv "$fixed" "$src"

    echo "[ok] replaced $src"
  done
done
```

## 5. 生成 compact 训练文件

`compact.jsonl` 是训练友好的结构化缓存，不重新跑 Qwen，不改变图片和裁剪偏好，只去掉冗余中间态并固定 `rig_targets`。

```bash
cd "$PROJECT"
export DATA=/root/autodl-tmp/data/beauty_dataset

for ds in cpc_rig gaic_rig; do
  for split in train val; do
    in_jsonl="$DATA/data/$ds/metadata/$split.jsonl"
    out_jsonl="$DATA/data/$ds/metadata/$split.compact.jsonl"

    [ -f "$in_jsonl" ] || continue

    python RIGCrop/scripts/compile_middle_state_train_jsonl.py \
      --input-jsonl "$in_jsonl" \
      --out-jsonl "$out_jsonl" \
      --max-nodes 12 \
      --overwrite
  done
done
```

检查：

```bash
ls -lh \
  "$DATA/data/cpc_rig/metadata/train.compact.jsonl" \
  "$DATA/data/cpc_rig/metadata/val.compact.jsonl" \
  "$DATA/data/gaic_rig/metadata/train.compact.jsonl"
```

如果存在 GAIC val，也检查：

```bash
ls -lh "$DATA/data/gaic_rig/metadata/val.compact.jsonl"
```

## 6. 无 GPU 数据读取验证

### 6.1 抽样检查图片存在

```bash
python - <<'PY'
import json
from pathlib import Path

files = [
    "/root/autodl-tmp/data/beauty_dataset/data/cpc_rig/metadata/train.compact.jsonl",
    "/root/autodl-tmp/data/beauty_dataset/data/cpc_rig/metadata/val.compact.jsonl",
    "/root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/train.compact.jsonl",
]

for fp in files:
    print("\n==", fp)
    missing = 0
    total = 0
    with open(fp, "r", encoding="utf-8") as f:
        for _, line in zip(range(200), f):
            r = json.loads(line)
            p = Path(r["image_path"])
            total += 1
            if not p.exists():
                missing += 1
                if missing <= 5:
                    print("missing:", p)
    print("checked", total, "missing", missing)
PY
```

通过标准：

```text
missing 0
```

### 6.2 Dataset batch 验证

当前 token-RoI 配置下，正式训练数据应该不返回 `winner_crop` / `loser_crop`。

```bash
PYTHONPATH=$PWD/RIGCrop python - <<'PY'
from torch.utils.data import DataLoader
from rigcrop.data import RIGPairwiseDataset

DATA = "/root/autodl-tmp/data/beauty_dataset"

common = dict(
    image_size=384,
    crop_size=224,
    max_records=16,
    max_pairs_per_record=8,
    max_nodes=12,
    return_crops=False,
)

datasets = {
    "cpc": RIGPairwiseDataset(
        f"{DATA}/data/cpc_rig/metadata/train.compact.jsonl",
        derive_pairs_from_scores=True,
        **common,
    ),
    "gaic": RIGPairwiseDataset(
        f"{DATA}/data/gaic_rig/metadata/train.compact.jsonl",
        derive_pairs_from_scores=True,
        min_score_gap=0.05,
        **common,
    ),
}

for name, ds in datasets.items():
    print("\n==", name)
    print("pairs", len(ds))
    batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
    print("keys", sorted(batch.keys()))
    print("image", batch["image"].shape)
    print("winner_box_feat", batch["winner_box_feat"].shape)
    print("loser_box_feat", batch["loser_box_feat"].shape)
    print("node_boxes", batch["node_boxes"].shape)
    assert "winner_crop" not in batch
    assert "loser_crop" not in batch
PY
```

通过标准：

```text
image torch.Size([2, 3, 384, 384])
winner_box_feat torch.Size([2, 8])
node_boxes torch.Size([2, 12, 4])
无 winner_crop / loser_crop
```

### 6.3 Config 实例化 Dataset

```bash
PYTHONPATH=$PWD/RIGCrop python - <<'PY'
import yaml
from rigcrop.data import RIGPairwiseDataset

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml"))

for i, d in enumerate(cfg["train_datasets"]):
    d = dict(d)
    d["max_records"] = 8
    ds = RIGPairwiseDataset(**d)
    print("train", i, "ok", len(ds), d["jsonl_path"], "return_crops", d.get("return_crops"))

vd = dict(cfg["val_dataset"])
vd["max_records"] = 8
val = RIGPairwiseDataset(**vd)
print("val ok", len(val), vd["jsonl_path"], "return_crops", vd.get("return_crops"))
PY
```

## 7. DINOv3 路径检查

不开 GPU 也可以先检查 repo 和权重文件是否存在。

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml"))
bb = cfg["model"]["backbone"]

for key in ["repo", "weights"]:
    p = Path(bb[key])
    print(key, p, "exists=", p.exists())

print("name", bb.get("name"))
print("source", bb.get("source"))
PY
```

通过标准：

```text
repo ... exists= True
weights ... exists= True
source local
name dinov3_vitb16
```

如果 `repo exists=False`，检查：

```bash
ls -lh /root/dinov3/hubconf.py
```

`repo` 必须指向包含 `hubconf.py` 的 DINOv3 代码目录，不是权重文件目录。

## 8. 开 GPU 后的单卡验证

启动 GPU 后先只用 0 卡验证。

### 8.1 CUDA 可见性

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/RIGCrop python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("gpu0", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

通过标准：

```text
cuda_available True
device_count 1
gpu0 NVIDIA GeForce RTX 4090
```

### 8.2 DINOv3 + encode_graph 验证

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/RIGCrop python - <<'PY'
import torch, yaml
from rigcrop.model import RIGCropModel

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml"))
model = RIGCropModel(**cfg["model"]).cuda().eval()

x = torch.randn(1, 3, 384, 384, device="cuda")

with torch.no_grad():
    g = model.encode_graph(x)

print("OK encode_graph")
print("visual_tokens", g["visual_tokens"].shape)
print("visual_spatial_size", g["visual_spatial_size"])
print("node_boxes", g["node_boxes"].shape)
print("relation_logits", g["relation_logits"].shape)
print("query_boxes", g["query_boxes"].shape)
print("mem_allocated_GB", torch.cuda.memory_allocated() / 1024**3)
print("mem_reserved_GB", torch.cuda.memory_reserved() / 1024**3)
PY
```

通过标准：

```text
visual_tokens torch.Size([1, 576, 384])
visual_spatial_size (24, 24)
node_boxes torch.Size([1, 12, 4])
relation_logits torch.Size([1, 12, 12, ...])
query_boxes torch.Size([1, 16, 4])
```

### 8.3 Token-RoI scoring 验证

这一步确认模型不需要 crop image，也能完成 winner/loser scoring。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/RIGCrop python - <<'PY'
import torch, yaml
from torch.utils.data import DataLoader
from rigcrop.data import RIGPairwiseDataset
from rigcrop.model import RIGCropModel

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml"))
d = dict(cfg["train_datasets"][0])
d["max_records"] = 4
d["max_pairs_per_record"] = 4
d["return_crops"] = False

ds = RIGPairwiseDataset(**d)
batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
assert "winner_crop" not in batch

model = RIGCropModel(**cfg["model"]).cuda().eval()
image = batch["image"].cuda(non_blocking=True)
winner_box = batch["winner_box_feat"].cuda(non_blocking=True)
loser_box = batch["loser_box_feat"].cuda(non_blocking=True)

with torch.no_grad():
    graph = model.encode_graph(image)
    pair_graph = {
        k: torch.cat([v, v], dim=0) if torch.is_tensor(v) and v.size(0) == image.size(0) else v
        for k, v in graph.items()
    }
    out = model(None, None, torch.cat([winner_box, loser_box], dim=0), graph=pair_graph)

print("OK token-roi scoring")
print("score", out["score"].shape)
print("utility", out["utility"].shape)
print("crop_state", out["crop_state"].shape)
PY
```

通过标准：

```text
OK token-roi scoring
score torch.Size([4])
utility torch.Size([4])
crop_state torch.Size([4, 384])
```

## 9. 单卡小训练冒烟

正式训练脚本没有 `max_steps` 参数。为了短跑验证，创建一个临时小配置，只取少量 records。

```bash
python - <<'PY'
from pathlib import Path
import yaml

src = Path("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml")
dst = Path("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.smoke_server.yaml")
cfg = yaml.safe_load(src.read_text())

cfg["output_dir"] = "RIGCrop/runs/server_smoke_token_roi"
cfg["epochs"] = 1
cfg["batch_size"] = 2
cfg["num_workers"] = 0
cfg["persistent_workers"] = False
cfg["prefetch_factor"] = 2
cfg["log_interval"] = 1

for d in cfg["train_datasets"]:
    d["max_records"] = 8
    d["max_pairs_per_record"] = 4
    d["return_crops"] = False

cfg["val_dataset"]["max_records"] = 4
cfg["val_dataset"]["max_pairs_per_record"] = 4
cfg["val_dataset"]["return_crops"] = False

dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(dst)
PY
```

启动单卡：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$PWD/RIGCrop \
python RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.smoke_server.yaml
```

通过标准：

```text
[rig-train] train_pairs=... val_pairs=... device=cuda:0 world_size=1
[rig-train-step] epoch=1 step=1/...
loss=...
```

如果 loss 不是 NaN，且能完成 1 epoch，就说明单卡训练链路正常。

## 10. 6 卡正式启动

正式启动前，建议换一个新的 `output_dir`，不要从旧的 crop-backbone checkpoint 继续训，因为当前 crop 表征路径已经从 crop image encoder 改成 token-RoI pooling。

### 10.1 检查没有残留训练进程

```bash
ps aux | grep train_rig_crop
nvidia-smi
```

如果有旧训练：

```bash
pkill -f "RIGCrop/scripts/train_rig_crop.py"
sleep 3
ps aux | grep train_rig_crop
```

### 10.2 启动 6 卡

```bash
cd "$PROJECT"
conda activate "$CONDA_ENV"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29521 \
NPROC=6 \
LOG_DIR=RIGCrop/logs \
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

脚本名虽然叫 `run_server_4gpu.sh`，但实际进程数由 `NPROC=6` 控制。

查看最新日志：

```bash
tail -f "$(ls -t RIGCrop/logs/rig_crop_train_*.log | head -1)"
```

通过标准：

```text
[run-server] nproc=6
[rig-train] train_pairs=... val_pairs=... device=cuda:0 world_size=6
[rig-train-step] epoch=1 step=1/...
```

## 11. 资源监控与 batch 判断

看 GPU：

```bash
nvitop
```

或：

```bash
watch -n 1 nvidia-smi
```

4090 24GB 的经验判断：

```text
显存 < 18GB：batch 可能还能上调
显存 18-21GB：比较舒服
显存 21-22.5GB：可跑，但不要再升 batch
显存 > 22.5GB：接近 OOM，建议降 batch
```

当前 Token-RoI 后，计算从约 `3 x DINOv3` 降到 `1 x DINOv3`，但总训练仍包含 entity decoder、relation head、crop graph attention、DDP 通信、dataloader，因此不会得到严格 3 倍加速。

## 12. 日志与训练产物

日志目录：

```bash
ls -lh RIGCrop/logs
tail -f "$(ls -t RIGCrop/logs/rig_crop_train_*.log | head -1)"
```

训练产物：

```bash
ls -lh RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth
```

应包含：

```text
best.pt
last.pt
history.json
history.png
```

查看训练曲线文件：

```bash
ls -lh RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/history.*
```

## 13. 常见错误处理

### 13.1 `ModuleNotFoundError: No module named 'rigcrop'`

在 repo 根目录执行，并加 `PYTHONPATH`：

```bash
cd "$PROJECT"
PYTHONPATH=$PWD/RIGCrop python - <<'PY'
import rigcrop
print("rigcrop ok")
PY
```

### 13.2 `ModuleNotFoundError: No module named 'cv2'`

```bash
python -m pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 13.3 `ModuleNotFoundError: No module named 'torchmetrics'`

DINOv3 local repo import 可能需要：

```bash
python -m pip install torchmetrics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 13.4 DINOv3 `hubconf.py` 找不到

检查：

```bash
ls -lh /root/dinov3/hubconf.py
```

配置中：

```yaml
repo: /root/dinov3
source: local
```

`repo` 不能写成 `.pth` 权重路径。

### 13.5 `changed=0` 且 `missing` 很多

说明 JSONL path rewrite 的 `OLD` 写错了。重新抽样：

```bash
python - <<'PY'
import json
p="/root/autodl-tmp/data/beauty_dataset/data/cpc_rig/metadata/val.jsonl"
with open(p) as f:
    for _, line in zip(range(5), f):
        print(json.loads(line)["image_path"])
PY
```

用抽样显示的公共前缀作为 `OLD`。

### 13.6 CUDA unknown error / 某张卡坏掉

先停训练：

```bash
pkill -f "RIGCrop/scripts/train_rig_crop.py"
```

检查：

```bash
nvidia-smi
```

如果某张卡 `Unknown Error`，不要把它放进 `CUDA_VISIBLE_DEVICES`。必要时重启服务器。

### 13.7 OOM

优先降：

```yaml
batch_size
```

然后降：

```yaml
num_workers
prefetch_factor
```

不要在显存已经超过 21GB 时继续升 batch。

### 13.8 `winner_crop` 缺失

如果使用新模式：

```yaml
crop_feature_mode: roi_tokens
return_crops: false
```

这是正确的。

如果你切回旧模式：

```yaml
crop_feature_mode: crop_backbone
```

必须同时设置：

```yaml
return_crops: true
```

## 14. 验证通过标准总表

| 阶段 | 命令/检查 | 通过标准 |
| --- | --- | --- |
| 依赖 | import torch/cv2/yaml | 无 import error |
| 语法 | `compileall` | 无输出 |
| 路径 | grep config | 无旧数据路径 |
| JSONL | image_path 抽样 | 指向当前服务器 |
| rewrite | stream rewrite | `missing=0` |
| compact | compile compact | 生成 `*.compact.jsonl` |
| Dataset | `return_crops=false` batch | 无 crop tensors，shape 正常 |
| DINOv3 | `encode_graph` | `visual_tokens` / `node_boxes` 正常 |
| Token-RoI | `model(None, None, box_feat, graph)` | score/utility 正常 |
| 单卡 | smoke train | 打印 step loss |
| 6 卡 | DDP train | `world_size=6`，step loss 正常 |
| 监控 | nvitop | GPU util 高，显存不过线 |

## 15. 正式训练前最后检查

```bash
grep -nE "jsonl_path|return_crops|crop_feature_mode|crop_pool_size|repo:|weights:|batch_size|num_workers|prefetch_factor|output_dir" \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

建议正式版至少满足：

```yaml
batch_size: 24
num_workers: 6
persistent_workers: true
prefetch_factor: 2
find_unused_parameters: false

model:
  crop_feature_mode: roi_tokens
  crop_pool_size: 4

loss:
  crop_pair: 1.0
  node: 0.1
  relation: 0.06
  utility: 0.15
  query: 0.05
  action: 0.0
```

如果 6 卡 4090 显存低于 20GB 且稳定跑过 1000 step，再考虑把 `batch_size` 从 24 提到 32。

