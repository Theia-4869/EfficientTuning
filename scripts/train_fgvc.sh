#!/bin/bash

gpu_id=0
model_root=models
data_path=(datasets/FGVC/CUB_200_2011 datasets/FGVC/nabirds datasets/FGVC/OxfordFlowers datasets/FGVC/StanfordDogs datasets/FGVC/StanfordCars)
output_dir=outputs/FGVC/subset/adamw
config_file=(cub nabirds flowers dogs cars)
percentile=(0.1 0.05 0.1 0.075 0.07)

for idx in {0..4}; do
    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
        --config-file configs/subset/${config_file[idx]}.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" \
        MODEL.SUBSET.PERCENTILE "${percentile[idx]}" \
        MODEL.SUBSET.LN_GRAD "True" \
        MODEL.SUBSET.WEIGHT_AND_BIAS "True" \
        MODEL.SUBSET.TYPE "layer" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        SOLVER.BASE_LR "0.005" \
        SOLVER.WEIGHT_DECAY "0.0001" \
        SOLVER.OPTIMIZER "adamw" \
        SEED "3407" \
        GPU_ID "${gpu_id}" \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path[idx]}" \
        OUTPUT_DIR "${output_dir}"
done
