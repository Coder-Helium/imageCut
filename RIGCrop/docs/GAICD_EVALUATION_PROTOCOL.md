# GAICD Evaluation Protocol for RIGCrop / RIGFormer

This note summarizes how to evaluate the current RIGCrop model on GAICD-style
candidate crops. It is written for paper reporting, not just for training
debugging.

## What GAICD Evaluates

GAICD provides many crop candidates per image, each with human MOS
(mean opinion score). A cropping model should score all candidates for one
image and return the best ranked crops. Therefore GAICD evaluation is a
candidate ranking problem, not only a pairwise preference problem.

The training log metric `pairwise_acc` in this repository is useful for
optimization, but it is not the GAICD official/table metric. It only checks
whether `score(winner_crop) > score(loser_crop)` for sampled training pairs.

## Main Metrics to Report

For an AAAI/CVPR-style composition paper, report these on GAICD:

1. **AccK/N**: return-K-of-top-N accuracy.
   - For each image, sort ground-truth crop candidates by MOS.
   - Let the GT positive set be the top `N` crops, usually `N in {5, 10}`.
   - Sort model predictions and take the top `K` crops, usually `K in {1,2,3,4}`.
   - `AccK/N` is the average fraction of predicted top-K crops that fall into
     the GT top-N set.

2. **Acc5 and Acc10**:
   - `Acc5 = mean(Acc1/5, Acc2/5, Acc3/5, Acc4/5)`.
   - `Acc10 = mean(Acc1/10, Acc2/10, Acc3/10, Acc4/10)`.
   - These are the most compact headline metrics for GAICD.

3. **SRCC**:
   - Spearman rank correlation between predicted crop scores and GT MOS.
   - This measures whether the model preserves the crop ordering.

4. **PCC**:
   - Pearson correlation between predicted crop scores and GT MOS.
   - This measures whether the predicted score scale is linearly aligned with
     human MOS.

Recommended main table columns:

```text
Method | Backbone | Acc1/5 | Acc2/5 | Acc3/5 | Acc4/5 | Acc5 |
       |          | Acc1/10 | Acc2/10 | Acc3/10 | Acc4/10 | Acc10 | SRCC | PCC
```

For a shorter ablation table:

```text
Method | Backbone | Acc5 | Acc10 | SRCC | PCC
```

## Secondary / Diagnostic Metrics

Use these for appendix or internal debugging:

- **Pairwise ranking accuracy**: checks all candidate pairs or sampled pairs.
  This is closer to CPC-style pairwise evaluation and the current training loss,
  but it is not a replacement for GAICD Acc5/Acc10.
- **Top-1 qualitative comparison**: save the predicted crop against the GT
  highest-MOS crop. This is useful for reviewer intuition but should not be the
  only quantitative evidence.
- **Latency and parameter count**: GAIC/CGS-style papers often compare runtime
  and model size. Report single-image inference time if you claim efficiency.

## Existing Research Convention

- GAIC introduced the GAICD benchmark and evaluates candidate crops with
  `AccK/N`, `Acc5`, `Acc10`, and SRCC. It sets `N` to 5 or 10 and evaluates
  `K=1,2,3,4`.
- Later methods such as CGS and Spatial-Aware Feature + Rank Consistency keep
  the same GAICD protocol and additionally report PCC in recent tables.
- Newer VLM-style cropping papers still report the same family of metrics on
  GAICD, often with `Acc1/5 ... Acc4/10`, `Acc5`, `Acc10`, `SRCC`, and `PCC`.

Useful references:

- GAIC / GAICD: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.pdf
- Spatial-Aware Feature and Rank Consistency: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Image_Cropping_With_Spatial-Aware_Feature_and_Rank_Consistency_CVPR_2023_paper.pdf
- CGS relation-based cropping: https://faculty.ucmerced.edu/mhyang/papers/cvpr2020_composing.pdf
- Cropper VLM cropping: https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Cropper_Vision-Language_Model_for_Image_Cropping_through_In-Context_Learning_CVPR_2025_paper.pdf

## Evaluation Script

Use:

```bash
cd /root/workspace/image_cut
conda activate myenv

mkdir -p RIGCrop/runs/gaicd_eval

PYTHONPATH=$PWD/RIGCrop \
CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/test_script/eval_gaicd_official_metrics.py \
  --jsonl /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/val.compact.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --batch-size 256 \
  --image-size 384 \
  --crop-size 224 \
  --out-json RIGCrop/runs/gaicd_eval/gaicd_val_metrics.json
```

Smoke test before the full run:

```bash
PYTHONPATH=$PWD/RIGCrop \
CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/test_script/eval_gaicd_official_metrics.py \
  --jsonl /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/val.compact.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --max-records 20 \
  --batch-size 256
```

If you also want the CPC-like diagnostic:

```bash
PYTHONPATH=$PWD/RIGCrop \
CUDA_VISIBLE_DEVICES=0 \
python RIGCrop/test_script/eval_gaicd_official_metrics.py \
  --jsonl /root/autodl-tmp/data/beauty_dataset/data/gaic_rig/metadata/val.compact.jsonl \
  --checkpoint RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt \
  --config RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml \
  --batch-size 256 \
  --compute-pairwise \
  --out-json RIGCrop/runs/gaicd_eval/gaicd_val_metrics_with_pairwise.json
```

## Interpreting Results

- For paper claims, prioritize `Acc5`, `Acc10`, `SRCC`, and `PCC`.
- If `Acc5/Acc10` improve but `SRCC/PCC` drop, the model finds acceptable top
  crops but does not rank the whole candidate set well.
- If `SRCC/PCC` improve but `Acc1/5` is low, the score ordering is broadly
  reasonable but the top crop selection is unstable.
- If only training `pairwise_acc` is high, do not claim GAICD SOTA. Validate on
  the full GAICD candidate set with this script.

