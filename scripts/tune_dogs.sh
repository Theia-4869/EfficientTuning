#!/bin/bash

gpu_id=4
model_root=models
data_path=datasets/FGVC/StanfordDogs
output_dir=outputs/FGVC/subset/tune_adamw

# rm logs/subset/dogs.log

CUDA_VISIBLE_DEVICES=${gpu_id} python tune_fgvc.py \
    --config-file configs/subset/dogs.yaml \
    --train-type "subset" \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "128" \
    MODEL.SUBSET.PERCENTILE "0.075" \
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
