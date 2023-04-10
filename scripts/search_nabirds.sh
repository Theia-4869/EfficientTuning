#!/bin/bash

gpu_id=1
model_root=models
data_path=datasets/FGVC/nabirds
output_dir=outputs/FGVC/subset/tune_percentile

# rm logs/subset/nabirds.log

CUDA_VISIBLE_DEVICES=${gpu_id} python search_fgvc.py \
    --config-file configs/subset/nabirds.yaml \
    --train-type "subset" \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "128" \
    MODEL.SUBSET.LN_GRAD "True" \
    MODEL.SUBSET.WEIGHT_AND_BIAS "True" \
    MODEL.SUBSET.TYPE "layer" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    SOLVER.OPTIMIZER "adamw" \
    SOLVER.BASE_LR "0.0005" \
    SOLVER.WEIGHT_DECAY "0.0001" \
    SEED "3407" \
    GPU_ID "${gpu_id}" \
    MODEL.MODEL_ROOT "${model_root}" \
    DATA.DATAPATH "${data_path}" \
    OUTPUT_DIR "${output_dir}"
