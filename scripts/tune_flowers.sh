#!/bin/bash

gpu_id=3
model_root=models
data_path=datasets/FGVC/OxfordFlowers
output_dir=outputs/FGVC/subset/tune_adamw

# rm logs/subset/flowers.log

CUDA_VISIBLE_DEVICES=${gpu_id} python tune_fgvc.py \
    --config-file configs/subset/flowers.yaml \
    --train-type "subset" \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "128" \
    MODEL.SUBSET.PERCENTILE "0.1" \
    MODEL.SUBSET.LN_GRAD "True" \
    MODEL.SUBSET.WEIGHT_AND_BIAS "True" \
    MODEL.SUBSET.TYPE "layer" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    SOLVER.OPTIMIZER "adamw" \
    SEED "3407" \
    GPU_ID "${gpu_id}" \
    MODEL.MODEL_ROOT "${model_root}" \
    DATA.DATAPATH "${data_path}" \
    OUTPUT_DIR "${output_dir}"
