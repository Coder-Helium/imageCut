# RIGFormer 新服务器迁移与训练 Runbook

这份文档汇总新服务器迁移时需要做的事情：环境安装、DINOv3 检查、JSONL 图片路径修复、数据读取测试、6 卡训练启动、日志查看和常见报错处理。

默认假设：

- 项目路径：`/home/hmx/workspace/imageCut`
- Conda 环境：`beauty_env`
- 数据根目录：`/home/hmx/nas/beauty_dataset`
- DINOv3 repo：`/home/hmx/dinov3`
- DINOv3 权重：`/home/hmx/nas/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- 主训练配置：`RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml`

如果新服务器用户名或路径不同，把下面命令里的 `/home/hmx/...` 替换成真实路径。

## 1. 拉取代码

推荐 SSH clone：

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:Coder-Helium/imageCut.git
cd imageCut
```

如果已经 clone 过：

```bash
cd /home/hmx/workspace/imageCut
git pull
```

确认当前路径存在，避免 pip 的 `os.getcwd()` 报错：

```bash
pwd
ls RIGCrop scripts requirements-rigformer.txt
```

## 2. 安装依赖

### 使用已有 conda 环境

```bash
conda activate beauty_env
cd /home/hmx/workspace/imageCut
CREATE_CONDA=0 ENV_NAME=beauty_env TORCH_CUDA=cu126 bash scripts/setup_rigformer_server.sh
```

如果 PyTorch/torchvision 已经装好，只安装项目其他依赖：

```bash
conda activate beauty_env
cd /home/hmx/workspace/imageCut
CREATE_CONDA=0 TORCH_CUDA=skip bash scripts/setup_rigformer_server.sh
```

如果服务器环境名是 `myenv`：

```bash
conda activate myenv
cd /root/workspace/imageCut
CREATE_CONDA=0 ENV_NAME=myenv TORCH_CUDA=skip bash scripts/setup_rigformer_server.sh
```

### 单独修复 torchvision/torch 不匹配

如果遇到：

```text
RuntimeError: operator torchvision::nms does not exist
```

重装匹配 CUDA 版本的 torchvision：

```bash
pip uninstall -y torchvision
pip install --index-url https://download.pytorch.org/whl/cu126 --force-reinstall torchvision
```

必要时一起重装 torch 和 torchvision：

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 --force-reinstall torch torchvision
```

## 3. 基础环境检查

```bash
cd /home/hmx/workspace/imageCut
conda activate beauty_env

python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
```

如果这里 `cuda_available=False`，先看：

```bash
nvidia-smi
```

如果 `nvidia-smi` 里某张卡 `Unknown Error`，先不要训练；排除坏卡或重启服务器后再测。

## 4. 配置 DINOv3 路径

确认 repo 和权重都存在：

```bash
ls /home/hmx/dinov3/hubconf.py
ls /home/hmx/nas/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

配置文件中应为：

```yaml
model:
  backbone:
    type: torchhub_dinov3
    repo: /home/hmx/dinov3
    source: local
    name: dinov3_vitb16
    pretrained: true
    weights: /home/hmx/nas/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

如果你的权重实际在 `/home/hmx/dinov3/...pth`，就把 `weights` 改成实际路径。

## 5. JSONL 图片路径修复

换服务器后，JSONL 里的 `image_path` 往往还是旧服务器路径。先检查当前 JSONL 的前几条：

```bash
cd /home/hmx/workspace/imageCut

python - <<'PY'
import json
paths = [
    "/home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl",
    "/home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl",
    "/home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.jsonl",
    "/home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/val.jsonl",
]
for p in paths:
    print("\n==", p)
    try:
        with open(p, "r", encoding="utf-8") as f:
            for i, line in zip(range(3), f):
                r = json.loads(line)
                print(r.get("sample_id"), r.get("image_path"))
    except FileNotFoundError:
        print("missing")
PY
```

### CPC 路径修复

如果 CPC 旧路径类似：

```text
/dmci_user/mx/ocean_nas/beauty_dataset/CPCDataset/images/xxx.jpg
```

新路径应为：

```text
/home/hmx/nas/beauty_dataset/CPCDataset/images/xxx.jpg
```

先 dry-run：

```bash
python RIGCrop/scripts/rewrite_jsonl_image_paths.py \
  --input-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl \
  --out-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.fixed.jsonl \
  --old-prefix /dmci_user/mx/ocean_nas/beauty_dataset \
  --new-prefix /home/hmx/nas/beauty_dataset \
  --check-exists \
  --dry-run

python RIGCrop/scripts/rewrite_jsonl_image_paths.py \
  --input-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl \
  --out-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.fixed.jsonl \
  --old-prefix /dmci_user/mx/ocean_nas/beauty_dataset \
  --new-prefix /home/hmx/nas/beauty_dataset \
  --check-exists \
  --dry-run
```

确认输出里：

```text
"changed": 9598
"missing": 0
```

或至少 `missing` 接近 0。确认无误后正式写出并替换：

```bash
cp /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl \
   /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl.bak_$(date +%Y%m%d_%H%M%S)
cp /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl \
   /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl.bak_$(date +%Y%m%d_%H%M%S)

python RIGCrop/scripts/rewrite_jsonl_image_paths.py \
  --input-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl \
  --out-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.fixed.jsonl \
  --old-prefix /dmci_user/mx/ocean_nas/beauty_dataset \
  --new-prefix /home/hmx/nas/beauty_dataset \
  --check-exists \
  --overwrite

python RIGCrop/scripts/rewrite_jsonl_image_paths.py \
  --input-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl \
  --out-jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.fixed.jsonl \
  --old-prefix /dmci_user/mx/ocean_nas/beauty_dataset \
  --new-prefix /home/hmx/nas/beauty_dataset \
  --check-exists \
  --overwrite

mv /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.fixed.jsonl \
   /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl
mv /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.fixed.jsonl \
   /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl
```

### GAIC 路径修复

先用第 5 节开头的检查命令看 GAIC 旧路径。如果旧路径也是某个固定前缀，把下面的 `OLD_GAIC_PREFIX` 换成真实前缀：

```bash
OLD_GAIC_PREFIX=/dmci_user/mx/ocean_nas/beauty_dataset
NEW_DATA_ROOT=/home/hmx/nas/beauty_dataset

python RIGCrop/scripts/rewrite_jsonl_image_paths.py \
  --input-jsonl ${NEW_DATA_ROOT}/data/gaic_rig/metadata/train.jsonl \
  --out-jsonl ${NEW_DATA_ROOT}/data/gaic_rig/metadata/train.fixed.jsonl \
  --old-prefix ${OLD_GAIC_PREFIX} \
  --new-prefix ${NEW_DATA_ROOT} \
  --check-exists \
  --dry-run

python RIGCrop/scripts/rewrite_jsonl_image_paths.py \
  --input-jsonl ${NEW_DATA_ROOT}/data/gaic_rig/metadata/val.jsonl \
  --out-jsonl ${NEW_DATA_ROOT}/data/gaic_rig/metadata/val.fixed.jsonl \
  --old-prefix ${OLD_GAIC_PREFIX} \
  --new-prefix ${NEW_DATA_ROOT} \
  --check-exists \
  --dry-run
```

如果 dry-run 没问题，再备份并替换：

```bash
cp ${NEW_DATA_ROOT}/data/gaic_rig/metadata/train.jsonl \
   ${NEW_DATA_ROOT}/data/gaic_rig/metadata/train.jsonl.bak_$(date +%Y%m%d_%H%M%S)
cp ${NEW_DATA_ROOT}/data/gaic_rig/metadata/val.jsonl \
   ${NEW_DATA_ROOT}/data/gaic_rig/metadata/val.jsonl.bak_$(date +%Y%m%d_%H%M%S)

python RIGCrop/scripts/rewrite_jsonl_image_paths.py \
  --input-jsonl ${NEW_DATA_ROOT}/data/gaic_rig/metadata/train.jsonl \
  --out-jsonl ${NEW_DATA_ROOT}/data/gaic_rig/metadata/train.fixed.jsonl \
  --old-prefix ${OLD_GAIC_PREFIX} \
  --new-prefix ${NEW_DATA_ROOT} \
  --check-exists \
  --overwrite

python RIGCrop/scripts/rewrite_jsonl_image_paths.py \
  --input-jsonl ${NEW_DATA_ROOT}/data/gaic_rig/metadata/val.jsonl \
  --out-jsonl ${NEW_DATA_ROOT}/data/gaic_rig/metadata/val.fixed.jsonl \
  --old-prefix ${OLD_GAIC_PREFIX} \
  --new-prefix ${NEW_DATA_ROOT} \
  --check-exists \
  --overwrite

mv ${NEW_DATA_ROOT}/data/gaic_rig/metadata/train.fixed.jsonl \
   ${NEW_DATA_ROOT}/data/gaic_rig/metadata/train.jsonl
mv ${NEW_DATA_ROOT}/data/gaic_rig/metadata/val.fixed.jsonl \
   ${NEW_DATA_ROOT}/data/gaic_rig/metadata/val.jsonl
```

如果 JSONL 里有 `rel_path` 字段，也可以用 `--image-root` 直接重建路径：

```bash
python RIGCrop/scripts/rewrite_jsonl_image_paths.py \
  --input-jsonl /home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.jsonl \
  --out-jsonl /home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.fixed.jsonl \
  --image-root /home/hmx/nas/beauty_dataset \
  --check-exists \
  --dry-run
```

## 6. JSONL 和 audit 质量检查

检查 RIG target 质量：

```bash
python - <<'PY'
import json
for p in [
    "/home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.audit.json",
    "/home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.audit.json",
]:
    print("\n", p)
    r = json.load(open(p))
    print(json.dumps(r["audit"]["rates"], indent=2, ensure_ascii=False))
    print("records", r["audit"]["records"])
    print("candidates", r["audit"]["total_candidates"])
    print("pairs", r["audit"]["total_pairwise_preferences"])
PY
```

检查图片路径是否存在：

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    "/home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl",
    "/home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl",
    "/home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.jsonl",
    "/home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/val.jsonl",
]
for p in paths:
    total = missing = 0
    examples = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            ip = r.get("image_path", "")
            if not Path(ip).exists():
                missing += 1
                if len(examples) < 5:
                    examples.append(ip)
    print("\n", p)
    print("total", total, "missing", missing)
    for e in examples:
        print("  missing:", e)
PY
```

## 7. DataLoader 测试

先测 CPC：

```bash
cd /home/hmx/workspace/imageCut
PYTHONPATH=$PWD/RIGCrop python - <<'PY'
from torch.utils.data import DataLoader
from rigcrop.data import RIGPairwiseDataset

ds = RIGPairwiseDataset(
    jsonl_path="/home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl",
    image_size=384,
    crop_size=224,
    max_records=16,
    max_pairs_per_record=8,
    max_nodes=12,
    derive_pairs_from_scores=True,
)
print("len", len(ds))
batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
for k in ["image", "winner_crop", "loser_crop", "winner_box_feat", "node_boxes", "relation_policy"]:
    print(k, batch[k].shape)
print("sample_id", batch["sample_id"][:2])
PY
```

再测 GAIC：

```bash
PYTHONPATH=$PWD/RIGCrop python - <<'PY'
from torch.utils.data import DataLoader
from rigcrop.data import RIGPairwiseDataset

ds = RIGPairwiseDataset(
    jsonl_path="/home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.jsonl",
    image_size=384,
    crop_size=224,
    max_records=16,
    max_pairs_per_record=8,
    max_nodes=12,
    derive_pairs_from_scores=True,
    min_score_gap=0.05,
)
print("len", len(ds))
batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
for k in ["image", "winner_crop", "loser_crop", "winner_box_feat", "node_boxes", "relation_policy"]:
    print(k, batch[k].shape)
print("sample_id", batch["sample_id"][:2])
PY
```

如果单 worker 没问题，再测多 worker：

```bash
PYTHONPATH=$PWD/RIGCrop python - <<'PY'
from torch.utils.data import ConcatDataset, DataLoader
from rigcrop.data import RIGPairwiseDataset

common = dict(image_size=384, crop_size=224, max_records=64, max_pairs_per_record=16, max_nodes=12, derive_pairs_from_scores=True)
cpc = RIGPairwiseDataset(jsonl_path="/home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl", **common)
gaic = RIGPairwiseDataset(jsonl_path="/home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.jsonl", min_score_gap=0.05, **common)
loader = DataLoader(ConcatDataset([cpc, gaic]), batch_size=6, num_workers=4, pin_memory=True)
batch = next(iter(loader))
print("OK mixed batch")
print("image", batch["image"].shape)
print("winner_crop", batch["winner_crop"].shape)
print("node_boxes", batch["node_boxes"].shape)
PY
```

## 8. DINOv3 模型前向测试

这一步测试 DINOv3 repo、pth 权重、RIGFormer 模型和 CUDA 前向是否都正常：

```bash
cd /home/hmx/workspace/imageCut
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/RIGCrop python - <<'PY'
import torch, yaml
from rigcrop.model import RIGCropModel

cfg = yaml.safe_load(open("RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml"))
cfg["model"]["backbone"]["repo"] = "/home/hmx/dinov3"
cfg["model"]["backbone"]["weights"] = "/home/hmx/nas/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

model = RIGCropModel(**cfg["model"]).cuda().eval()
x = torch.randn(1, 3, 384, 384, device="cuda")
with torch.no_grad():
    g = model.encode_graph(x)

print("OK")
print("node_boxes", g["node_boxes"].shape)
print("relation_logits", g["relation_logits"].shape)
print("query_boxes", g["query_boxes"].shape)
print("gpu_mem_allocated_GB", torch.cuda.memory_allocated() / 1024**3)
print("gpu_mem_reserved_GB", torch.cuda.memory_reserved() / 1024**3)
PY
```

期望形状：

```text
node_boxes torch.Size([1, 12, 4])
relation_logits torch.Size([1, 12, 12, 6])
query_boxes torch.Size([1, 16, 4])
```

如果报 `ModuleNotFoundError: No module named 'torchmetrics'`：

```bash
pip install torchmetrics
```

或重新跑依赖安装脚本。

## 9. 训练配置检查

主配置建议：

```yaml
batch_size: 6
num_workers: 4
epochs: 30
log_interval: 1000
ddp_timeout_seconds: 7200
```

路径必须是新服务器绝对路径：

```yaml
train_datasets:
  - jsonl_path: /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/train.jsonl
  - jsonl_path: /home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/train.jsonl

val_dataset:
  jsonl_path: /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl
```

说明：

- `batch_size` 是每张 GPU 的 batch。6 卡、`batch_size: 6` 等价于全局 batch 36。
- `num_workers: 4` 是每个 DDP 进程 4 个 DataLoader worker。6 卡总共 24 个 worker。
- 2080 11GB 上建议 `batch_size: 6` 起步；如果显存约 7.5GB/11GB 且稳定，可以试 8。
- 3090 24GB 上可以从 6 或 8 起步，再看显存和吞吐。

## 10. 启动训练

### 6 卡训练

```bash
cd /home/hmx/workspace/imageCut
conda activate beauty_env

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29523 \
NPROC=6 \
LOG_DIR=RIGCrop/logs \
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

脚本名虽然叫 `run_server_4gpu.sh`，实际卡数由 `NPROC` 控制。

### 指定 2,3,4,5 四张卡

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29516 \
NPROC=4 \
LOG_DIR=RIGCrop/logs \
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

注意不要在命令最前面写反斜杠。错误示例：

```bash
\MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 ...
```

这会报：

```text
MASTER_ADDR=127.0.0.1: command not found
```

### 从断点恢复训练

训练脚本会保存：

```text
RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/last.pt
RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt
```

恢复上次训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29524 \
NPROC=6 \
LOG_DIR=RIGCrop/logs \
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --resume RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/last.pt
```

也可以在配置里开启自动恢复：

```yaml
auto_resume: true
```

手动 `--resume` 更稳，避免误用旧输出目录里的 `last.pt`。

## 11. 查看日志和进度

启动脚本会打印日志路径，例如：

```text
[run-server] log=RIGCrop/logs/rig_crop_train_20260629_164544.log
```

查看：

```bash
tail -f RIGCrop/logs/rig_crop_train_20260629_164544.log
```

正常训练会打印：

```text
[rig-train] train_pairs=1361024 val_pairs=131584 device=cuda:0 world_size=6
[rig-train-step] epoch=1 step=1000/37807 loss=... acc=... eta=...
```

每 `log_interval: 1000` step 打印一次。`step=1` 的 ETA 不准，要看 `step=1000` 之后。

查看 GPU：

```bash
nvitop
# 或
watch -n 2 nvidia-smi
```

查看进程：

```bash
ps aux | grep train_rig_crop
```

## 12. 停止训练

先查进程：

```bash
ps aux | grep train_rig_crop
```

温和停止：

```bash
pkill -u "$USER" -f "RIGCrop/scripts/train_rig_crop.py.*rig_crop_cpc_gaic_dinov3_pth.yaml"
```

如果还有 torchrun 父进程：

```bash
pkill -u "$USER" -f "torchrun.*train_rig_crop.py.*rig_crop_cpc_gaic_dinov3_pth.yaml"
```

再次确认：

```bash
ps aux | grep train_rig_crop
```

只剩 `grep train_rig_crop` 就说明停干净了。

## 13. 输出文件

训练输出目录：

```text
RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/
```

关键文件：

```text
best.pt
last.pt
history.json
training_curves.png
loss.png
pairwise_acc.png
score_margin.png
node_loss.png
relation_loss.png
utility_loss.png
query_loss.png
action_loss.png
```

`best.pt` 是验证集 pairwise accuracy 最好的权重，后续预测和论文评测优先用它。

## 14. 论文评测命令

### Pairwise preference

CPC 或 GAIC 都可以跑 pairwise accuracy：

```bash
PYTHONPATH=$PWD/RIGCrop python RIGCrop/scripts/eval_rig_crop.py \
  --jsonl /home/hmx/nas/beauty_dataset/data/cpc_rig/metadata/val.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --batch-size 32
```

输出指标：

```text
pairwise_acc
weighted_pairwise_acc
mean_score_margin
```

### Candidate ranking

GAIC 更适合报告候选集排序指标：

```bash
PYTHONPATH=$PWD/RIGCrop python RIGCrop/scripts/eval_rig_crop_candidates.py \
  --jsonl /home/hmx/nas/beauty_dataset/data/gaic_rig/metadata/val.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --batch-size 64 \
  --acc-k 1,5,10 \
  --out-json RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/gaic_candidate_eval.json
```

输出指标：

```text
srcc_global
srcc_mean_per_image
acc_at_k
```

论文主表建议至少包含：

- CPC：pairwise accuracy / weighted pairwise accuracy
- GAIC：SRCC / Acc@1 / Acc@5
- 消融：crop-only、no-relation、no-importance、no-utility
- 效率：每图候选排序时间、显存峰值

## 15. 常见问题

### 找不到 JSONL

报错：

```text
FileNotFoundError: data/cpc_rig/metadata/train.jsonl
```

原因：配置还是旧相对路径。把 YAML 里的 `jsonl_path` 改成 `/home/hmx/nas/beauty_dataset/...` 绝对路径。

### 找不到 rigcrop

报错：

```text
ModuleNotFoundError: No module named 'rigcrop'
```

测试脚本前加：

```bash
PYTHONPATH=$PWD/RIGCrop
```

训练脚本内部已自动插入路径，一般不会遇到。

### CUDA 初始化失败

报错：

```text
CUDA initialization: CUDA unknown error
Can't initialize NVML
```

先看：

```bash
nvidia-smi
```

如果某张卡 `Unknown Error`，排除该卡：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

如果排除坏卡仍不行，通常需要管理员重置 GPU 或重启服务器。

### epoch 结束后 NCCL timeout

现象：

```text
Watchdog caught collective operation timeout
barrier from train_rig_crop.py
```

旧代码里验证只在 rank0 单卡跑，其他 rank 在 barrier 等太久会超时。解决：

- 同步最新 `RIGCrop/scripts/train_rig_crop.py`
- 确认配置里有 `ddp_timeout_seconds: 7200`
- 重新启动训练

新代码已经改为分布式验证，6 张卡会一起算 val。

### pip 报 os.getcwd FileNotFoundError

报错：

```text
FileNotFoundError: [Errno 2] No such file or directory
os.getcwd()
```

说明当前 shell 所在目录失效了。处理：

```bash
cd /home/hmx/workspace/imageCut
python -m pip --version
CREATE_CONDA=0 TORCH_CUDA=skip bash scripts/setup_rigformer_server.sh
```

最新安装脚本已经对 pip 调用做了 `cd "${ROOT_DIR}"` 保护。

### DDP find_unused_parameters 警告

日志：

```text
find_unused_parameters=True was specified ... did not find any unused parameters
```

这是性能警告，不是错误。当前主配置已经使用 `find_unused_parameters: false`，如果仍看到这条，说明你跑的是旧代码或旧配置。

## 16. 训练是否健康的快速判断

健康日志通常是：

```text
crop loss 下降
pairwise acc 上升
score margin 上升
```

例如：

```text
step=1000  crop=0.5921 acc=0.6423 margin=0.1943
step=37807 crop=0.5422 acc=0.7107 margin=0.2929
```

这说明主裁剪排序任务在学习。`node / relation / utility / query` 是辅助弱监督，不要只看这些辅助 loss 判断最终裁剪效果。
