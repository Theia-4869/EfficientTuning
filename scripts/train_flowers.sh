# launch final training for FGVC-flowers. The hyperparameters are the same from our paper.
gpu_id=3
model_root=models
data_path=datasets/FGVC/OxfordFlowers
output_dir=outputs/FGVC/test/gradient_head

# rm -rf ${output_dir}

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-file configs/subset/flowers.yaml \
    DATA.BATCH_SIZE "128" \
    DATA.DATAPATH "${data_path}" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    GPU_ID "${gpu_id}" \
    MODEL.MODEL_ROOT "${model_root}" \
    MODEL.SUBSET.LEVEL "block" \
    MODEL.SUBSET.LN_GRAD "True" \
    MODEL.SUBSET.PERCENTILE "0.1" \
    MODEL.SUBSET.SELECT "gradient" \
    MODEL.SUBSET.WEIGHT_AND_BIAS "False" \
    MODEL.TYPE "vit" \
    METHOD.CONTRASTIVE "False" \
    OUTPUT_DIR "${output_dir}" \
    SEED "3407" \
    SOLVER.BASE_LR "0.005" \
    SOLVER.WEIGHT_DECAY "0.0001" \
    SOLVER.OPTIMIZER "adamw"
