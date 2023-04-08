#!/bin/bash

gpu_id=6
model_root=models
data_path=datasets/FGVC/StanfordCars
output_dir=outputs/FGVC/subset/test

rm -rf ${output_dir}

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-file configs/subset/cars.yaml \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "128" \
    MODEL.SUBSET.PERCENTILE "0.05" \
    MODEL.SUBSET.LN_GRAD "True" \
    MODEL.SUBSET.WEIGHT_AND_BIAS "True" \
    MODEL.SUBSET.TYPE "layer" \
    MODEL.SUBSET.MODE "" \
    MODEL.SUBSET.REINITIALIZE "False" \
    MODEL.SUBSET.REINITIALIZE_TYPE "constant" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    SOLVER.BASE_LR "0.05" \
    SOLVER.WEIGHT_DECAY "0.0001" \
    SEED "3407" \
    GPU_ID "${gpu_id}" \
    MODEL.MODEL_ROOT "${model_root}" \
    DATA.DATAPATH "${data_path}" \
    OUTPUT_DIR "${output_dir}"