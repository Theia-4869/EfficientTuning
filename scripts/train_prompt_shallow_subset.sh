#!/bin/bash

gpu_id=6
model_root=models
data_path=(datasets/FGVC/CUB_200_2011 datasets/FGVC/nabirds datasets/FGVC/OxfordFlowers datasets/FGVC/StanfordDogs datasets/FGVC/StanfordCars)
output_dir=outputs/FGVC/prompt/shallow_subset
config_file=(cub nabirds flowers dogs cars)
prompt_num=(100 50 100 100 100)
percentile=(0.1 0.05 0.1 0.075 0.07)
lr=(0.5 10.0 5.0 1.0 500.0)
wd=(0.0001 0.0 0.0001 0.0001 0.0)

for idx in {0..4}; do
    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
        --config-file configs/prompt/${config_file[idx]}.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" \
        MODEL.PROMPT.DEEP "False" \
        MODEL.PROMPT.NUM_TOKENS "${prompt_num[idx]}" \
        MODEL.PROMPT.DROPOUT "0.0" \
        MODEL.SUBSET.PERCENTILE "0.05" \
        MODEL.SUBSET.LN_GRAD "True" \
        MODEL.SUBSET.WEIGHT_AND_BIAS "True" \
        MODEL.SUBSET.TYPE "layer" \
        MODEL.SUBSET.MODE "prompt" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        SOLVER.BASE_LR "${lr[idx]}" \
        SOLVER.WEIGHT_DECAY "${wd[idx]}" \
        SEED "3407" \
        GPU_ID "${gpu_id}" \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path[idx]}" \
        OUTPUT_DIR "${output_dir}"
done
