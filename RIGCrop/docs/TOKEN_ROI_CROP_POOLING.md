# Token-RoI Crop Pooling Optimization

## 1. Motivation

The previous DINOv3 RIGFormer training path encoded each pairwise sample with three visual backbone passes:

```text
full image -> DINOv3 -> graph/entity/relation state
winner crop -> DINOv3 -> winner crop state
loser crop -> DINOv3 -> loser crop state
```

For a per-GPU batch size `B`, one training step therefore executed approximately:

```text
B full-image backbone passes + 2B crop backbone passes
```

With DINOv3 ViT-B this is expensive because the backbone dominates both compute and activation memory. The optimized path changes crop scoring to a single-pass image-conditioned design:

```text
full image -> DINOv3 patch tokens -> graph/entity/relation state
                                      -> token-space RoI pooling for each crop
                                      -> graph-aware crop scoring transformer
```

The backbone now runs once per original image. Winner and loser crops are represented by pooled full-image tokens instead of separate crop images.

## 2. Method

### 2.1 Full-image token cache inside graph state

`RIGCropModel.encode_graph(image)` now stores the visual token map in the graph dictionary:

```python
{
    "visual_tokens": visual.tokens,
    "visual_spatial_size": visual.spatial_size,
    "node_tokens": ...,
    "relation_logits": ...,
}
```

For DINOv3 ViT-B at `384 x 384` with patch size 16, this is typically:

```text
visual_tokens: B x 576 x D
visual_spatial_size: (24, 24)
```

### 2.2 Token-space RoI pooling

For each candidate crop box `[x1, y1, x2, y2]` in normalized coordinates, the model:

1. reshapes `visual_tokens` into `B x D x Hf x Wf`;
2. creates a small regular grid inside the candidate box;
3. samples the token map with bilinear `grid_sample`;
4. averages the sampled grid into one crop token.

The crop token then replaces the old `crop_visual.pooled` vector:

```text
crop_token =
    roi_pooled_visual_token
  + box_geometry_embedding
  + graph_utility_component_embedding
```

The rest of the graph-aware crop transformer is unchanged.

### 2.3 Config switches

The optimized mode is controlled by:

```yaml
model:
  crop_feature_mode: roi_tokens
  crop_pool_size: 4
```

Supported aliases:

```text
roi_tokens, token_roi, token_pool, token_pooling, single_pass
```

The legacy behavior is still available:

```yaml
model:
  crop_feature_mode: crop_backbone
```

Legacy mode requires `winner_crop` and `loser_crop` tensors and runs the backbone on crop images.

## 3. Data Pipeline Change

Because `roi_tokens` does not need explicit crop images during training, the DINOv3 config now sets:

```yaml
return_crops: false
```

for CPC train, GAIC train, and validation datasets.

This means the Dataset still returns:

```text
image
winner_box_feat
loser_box_feat
node/relation/query supervision
pairwise weights
```

but skips:

```text
winner_crop
loser_crop
```

This reduces CPU-side crop extraction, host memory pressure, dataloader transfer volume, and GPU input tensor memory.

Default Dataset behavior remains backward compatible:

```yaml
return_crops: true
```

if the field is omitted.

## 4. Expected Compute Impact

The dominant backbone work changes from:

```text
3 visual backbone passes per pairwise sample
```

to:

```text
1 visual backbone pass per pairwise sample
```

The total speedup will be smaller than 3x because training still includes:

```text
entity decoder
relation pairwise head
crop graph attention
loss computation
DDP communication
dataloader image read/resize
validation
```

But this is the most important architectural compute optimization before adding AMP or feature caching.

## 5. Paper-facing Interpretation

This change is methodologically cleaner than repeatedly encoding crop images. The model can be described as:

```text
single-pass image-conditioned crop reasoning
```

or:

```text
token-space candidate crop pooling over a foundation visual backbone
```

The crop score is conditioned on the same global image representation that produced the entity-relation graph. This makes the method closer to modern detection/segmentation/query-based transformer pipelines:

```text
image tokens -> structured graph state -> candidate query/crop state -> score
```

This is a stronger AAAI-facing story than treating every crop as an independent image.

## 6. Updated Training Path

In training, `_score_pair` now checks:

```python
model.uses_crop_image()
```

If false, it passes:

```python
image_arg = None
crop_arg = None
```

and only sends the concatenated winner/loser box features plus the duplicated graph state:

```text
pair_box_feat: 2B x 8
pair_graph.visual_tokens: 2B x N x D
```

The model then obtains winner/loser crop states from the already-computed image tokens.

## 7. Updated Prediction Path

`predict_rig_crop.py` and `eval_rig_crop_candidates.py` also respect `uses_crop_image()`:

```text
roi_tokens mode:
  do not crop candidate images
  score candidates from full-image tokens

crop_backbone mode:
  preserve old crop-image scoring
```

This keeps training and inference consistent.

## 8. Validation Checklist

Run a CPU smoke compile first:

```bash
python -m compileall -q RIGCrop/rigcrop RIGCrop/scripts
```

Run the smoke pipeline:

```bash
bash RIGCrop/scripts/run_smoke_test.sh
```

For DINOv3 server training, verify the config contains:

```yaml
model:
  crop_feature_mode: roi_tokens
  crop_pool_size: 4

train_datasets:
  - return_crops: false
  - return_crops: false

val_dataset:
  return_crops: false
```

Then start training as usual:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29521 \
NPROC=6 \
LOG_DIR=RIGCrop/logs \
bash RIGCrop/scripts/run_server_4gpu.sh \
  RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml
```

## 9. Reverting to Legacy Crop Encoding

If an ablation needs the old behavior, set:

```yaml
model:
  crop_feature_mode: crop_backbone
```

and make sure datasets return crop tensors:

```yaml
return_crops: true
```

This restores the previous three-backbone-pass pipeline.

## 10. Caveats

Token-RoI pooling samples from the resized full-image feature map, not from a separately resized crop image. This is computationally better and architecturally more coherent, but it changes the inductive bias:

```text
old: crop-specific resized visual detail
new: crop region represented in global image-token context
```

For aesthetics-aware cropping, the new bias is usually preferable because crop quality depends on global composition and relation preservation. Very tiny crops may lose fine-grained detail compared with independent crop encoding, but the candidate boxes in CPC/GAIC are generally composition-level crops rather than object-detection micro boxes.

