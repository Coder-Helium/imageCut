# GAICD-Optimized Training With Middle-State Supervision

This document explains the GAICD-focused training upgrade added after observing:

```text
CPC pairwise_acc ~= 0.761
GAICD Acc5      ~= 0.288
GAICD Acc10     ~= 0.466
GAICD SRCC      ~= 0.710
```

The model already learns useful composition preference and preserves semantic
integrity well, but GAICD `Acc5/Acc10` requires the top MOS candidates to be
retrieved very accurately. The old training pipeline converted GAICD MOS into
sampled pairs and mixed it with a much larger CPC pair set, which weakens the
GAICD top-k signal.

## What Changed

### 1. GAICD Candidate-List Dataset

File:

```text
RIGCrop/rigcrop/data.py
```

New components:

```python
RIGCandidateListDataset
collate_candidate_lists
```

Instead of sampling one winner/loser pair, this dataset returns all scored
candidate crops for one image:

```text
image
candidate_box_feats: B x C x 8
candidate_scores:    B x C
candidate_mask:      B x C
middle-state targets
```

This preserves the native GAICD supervision:

```text
one image -> many candidates -> human MOS ranking
```

### 2. GAICD Listwise Loss

File:

```text
RIGCrop/rigcrop/losses.py
```

New losses:

```python
listwise_crop_loss
topk_hard_negative_loss
```

`listwise_crop_loss` uses a ListNet-style objective:

```text
target_probs = softmax(MOS / temperature)
loss = CE(target_probs, log_softmax(predicted_scores))
```

This optimizes full candidate ranking and helps `SRCC/PCC`.

`topk_hard_negative_loss` compares:

```text
hard positive = lowest predicted score among GT MOS top5
hard negative = highest predicted score outside GT MOS top10
```

and pushes:

```text
score(top5 positive) > score(hard negative)
```

This directly targets `Acc5/Acc10`.

### 3. Pairwise + Listwise Joint Training

File:

```text
RIGCrop/scripts/train_rig_crop.py
```

The original branch is still preserved:

```text
CPC/GAIC pairwise batch
-> pairwise_crop_loss
-> node/relation/utility/query/action middle-state losses
```

The new branch runs alongside it:

```text
GAICD candidate-list batch
-> listwise_crop_loss
-> topk_hard_negative_loss
```

The total loss is now:

```text
total =
  crop_pair * pairwise_loss
+ middle_state_losses
+ gaic_listwise * listwise_loss
+ gaic_topk * topk_hard_negative_loss
```

Important: middle-state weights are not reduced. They remain the method's core
supervision and continue to train the interpretable composition state.

### 4. GAICD Metric-Aware Checkpointing

The old `best.pt` was selected by `val_pairwise_acc`, which favors CPC-style
pairwise ranking. The upgraded config uses:

```yaml
best_metric: val_gaic_acc10
```

Training now saves:

```text
best.pt          best by configured metric, now GAICD Acc10
best_gaic.pt     best epoch when GAICD metrics exist
best_pairwise.pt best by CPC/val pairwise accuracy
last.pt          latest epoch
```

Each epoch can also log:

```text
val_gaic_acc1_5
val_gaic_acc2_5
val_gaic_acc3_5
val_gaic_acc4_5
val_gaic_acc5
val_gaic_acc1_10
val_gaic_acc2_10
val_gaic_acc3_10
val_gaic_acc4_10
val_gaic_acc10
val_gaic_srcc
val_gaic_pcc
```

### 5. GAICD Pairwise Rebalancing

The original mixed dataset was approximately:

```text
CPC : GAICD ~= 9 : 1
```

The config now supports per-dataset repeat:

```yaml
train_datasets:
  - CPC
  - GAICD
    repeat: 8
```

This increases GAICD pairwise exposure while the listwise branch supplies the
main GAICD top-k signal.

## Updated Config Fields

The main config now includes:

```yaml
best_metric: val_gaic_acc10

gaic_listwise_batch_size: 1
gaic_val_batch_size: 1
gaic_listwise_every: 1
gaic_eval_interval: 1

gaic_listwise_train_dataset:
  jsonl_path: /path/to/data/gaic_rig/metadata/train.compact.jsonl
  image_size: 384
  max_nodes: 12
  compact_records: true
  image_cache_size: 8

gaic_listwise_val_dataset:
  jsonl_path: /path/to/data/gaic_rig/metadata/val.compact.jsonl
  image_size: 384
  max_nodes: 12
  compact_records: true
  image_cache_size: 8

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

For 6 x 4090, keep:

```yaml
gaic_listwise_batch_size: 1
```

The candidate-list branch scores about 90 candidates per image. Larger
`gaic_listwise_batch_size` may improve throughput but increases memory sharply.

## Recommended GAICD Fine-Tuning Flow

Do not simply continue the old concat-pairwise run. Start a GAICD-focused
fine-tune from the best available checkpoint:

```bash
cd /root/workspace/image_cut
conda activate myenv

# Edit paths in the config first if needed:
# - CPC train/val compact jsonl
# - GAICD train/val compact jsonl
# - DINOv3 repo and weight path
# - output_dir

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29531 \
NPROC=6 \
LOG_DIR=RIGCrop/logs \
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

If using `torchrun` directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --nproc_per_node=6 \
  --master_addr=127.0.0.1 \
  --master_port=29531 \
  RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --resume RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt
```

For a new run directory, set:

```yaml
output_dir: RIGCrop/runs/rig_crop_cpc_gaic_dinov3_gaic_opt
```

Then resume from the old run:

```bash
--resume RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt
```

## Smoke Test Before Full Training

Create a temporary smoke config by setting:

```yaml
batch_size: 2
gaic_listwise_batch_size: 1
num_workers: 0
epochs: 1
log_interval: 1

train_datasets:
  - max_records: 8
  - max_records: 8

gaic_listwise_train_dataset:
  max_records: 8

gaic_listwise_val_dataset:
  max_records: 8

val_dataset:
  max_records: 8
```

Then run one GPU:

```bash
PYTHONPATH=$PWD/RIGCrop \
CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.smoke.yaml \
  --resume RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt
```

Expected log fields:

```text
gaic_total
gaic_lw
gaic_topk
gaic_acc5
gaic_acc10
```

Expected epoch metrics:

```text
val_gaic_acc5
val_gaic_acc10
val_gaic_srcc
val_gaic_pcc
```

## What to Watch

Primary GAICD metrics:

```text
val_gaic_acc5
val_gaic_acc10
val_gaic_srcc
val_gaic_pcc
```

Primary CPC metrics:

```text
val_pairwise_acc
val_score_margin
```

Middle-state stability:

```text
val_node_loss
val_relation_loss
val_utility_loss
```

Good training behavior:

```text
val_gaic_acc10 increases
val_gaic_acc5 increases
val_gaic_srcc/PCC do not collapse
val_pairwise_acc does not fall sharply
middle-state losses stay bounded
```

Bad training behavior:

```text
val_gaic_acc10 flat
val_node_loss / val_relation_loss explode
val_pairwise_acc drops sharply
gaic_topk loss stays high
```

If memory is too high:

```yaml
gaic_listwise_batch_size: 1
batch_size: reduce pairwise batch
prefetch_factor: 2
```

If GAICD metrics rise but CPC drops too much:

```yaml
train_datasets:
  - CPC repeat: 1
  - GAICD repeat: 4
loss:
  gaic_listwise: 0.3
  gaic_topk: 0.4
```

If GAICD metrics do not rise:

```yaml
loss:
  gaic_listwise: 0.7
  gaic_topk: 0.7
gaic_listwise_temperature: 0.25
```

Do not lower the middle-state weights unless an ablation specifically tests
that choice, because the paper's core claim depends on the structured
middle-state bottleneck.

