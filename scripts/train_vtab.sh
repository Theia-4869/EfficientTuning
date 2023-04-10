#!/bin/bash

gpu_id=0
model_root=models
dataset=("vtab-cifar(num_classes=100)" "vtab-caltech101" "vtab-dtd" "vtab-oxford_flowers102" "vtab-oxford_iiit_pet" "vtab-svhn" "vtab-sun397" "vtab-patch_camelyon" "vtab-eurosat" "vtab-resisc45" 'vtab-diabetic_retinopathy(config="btgraham-300")' 'vtab-clevr(task="count_all")' 'vtab-clevr(task="closest_object_distance")' "vtab-dmlab" 'vtab-kitti(task="closest_vehicle_distance")' 'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)' 'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)' 'vtab-smallnorb(predicted_attribute="label_azimuth")' 'vtab-smallnorb(predicted_attribute="label_elevation")')
number_classes=(100 102 47 102 37 10 397 2 10 45 5 8 6 6 4 16 16 18 9)
output_dir=outputs/VTAB/subset/test

for idx in {0..18}; do
    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
        --config-file configs/subset/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" \
        MODEL.SUBSET.PERCENTILE "0.05" \
        MODEL.SUBSET.LN_GRAD "True" \
        MODEL.SUBSET.WEIGHT_AND_BIAS "True" \
        MODEL.SUBSET.TYPE "layer" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME ${dataset[idx]} \
        DATA.NUMBER_CLASSES "${number_classes[idx]}" \
        SOLVER.BASE_LR "0.005" \
        SOLVER.WEIGHT_DECAY "0.0001" \
        SOLVER.OPTIMIZER "adamw" \
        SEED "3407" \
        GPU_ID "${gpu_id}" \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "datasets/VTAB" \
        OUTPUT_DIR "${output_dir}"
done
