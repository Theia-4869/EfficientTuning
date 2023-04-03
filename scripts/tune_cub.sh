# launch final training for FGVC-cub. The hyperparameters are the same from our paper.
gpu_id=5
model_root=models
data_path=datasets/FGVC/CUB2002011/CUB_200_2011
output_dir=outputs/FGVC/cub/subset/LN_true_reinit_false

# CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
#     --config-file configs/prompt/cub.yaml \
#     MODEL.TYPE "vit" \
#     DATA.BATCH_SIZE "128" \
#     MODEL.PROMPT.NUM_TOKENS "10" \
#     MODEL.PROMPT.DEEP "True" \
#     MODEL.PROMPT.DROPOUT "0.1" \
#     DATA.FEATURE "sup_vitb16_imagenet21k" \
#     SOLVER.BASE_LR "10.0" \
#     SOLVER.WEIGHT_DECAY "0.001" \
#     SEED "3407" \
#     GPU_ID "${gpu_id}" \
#     MODEL.MODEL_ROOT "${model_root}" \
#     DATA.DATAPATH "${data_path}" \
#     OUTPUT_DIR "${output_dir}"

# CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
#     --config-file configs/prompt/cub.yaml \
#     MODEL.TYPE "vit" \
#     DATA.BATCH_SIZE "64" \
#     MODEL.PROMPT.NUM_TOKENS "100" \
#     MODEL.PROMPT.DEEP "False" \
#     MODEL.PROMPT.DROPOUT "0.0" \
#     DATA.FEATURE "sup_vitb16_imagenet21k" \
#     SOLVER.BASE_LR "0.5" \
#     SOLVER.WEIGHT_DECAY "0.001" \
#     SEED "3407" \
#     GPU_ID "${gpu_id}" \
#     MODEL.MODEL_ROOT "${model_root}" \
#     DATA.DATAPATH "${data_path}" \
#     OUTPUT_DIR "${output_dir}"

# CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
#     --config-file configs/linear/cub.yaml \
#     MODEL.TYPE "vit" \
#     DATA.BATCH_SIZE "2048" \
#     DATA.FEATURE "sup_vitb16_imagenet21k" \
#     SOLVER.BASE_LR "5.0" \
#     SOLVER.WEIGHT_DECAY "0.0001" \
#     SEED "3407" \
#     GPU_ID "${gpu_id}" \
#     MODEL.MODEL_ROOT "${model_root}" \
#     DATA.DATAPATH "${data_path}" \
#     OUTPUT_DIR "${output_dir}"

# CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
#     --config-file configs/finetune/cub.yaml \
#     MODEL.TYPE "vit" \
#     DATA.BATCH_SIZE "128" \
#     DATA.FEATURE "sup_vitb16_imagenet21k" \
#     SOLVER.BASE_LR "0.005" \
#     SOLVER.WEIGHT_DECAY "0.0" \
#     SEED "3407" \
#     GPU_ID "${gpu_id}" \
#     MODEL.MODEL_ROOT "${model_root}" \
#     DATA.DATAPATH "${data_path}" \
#     OUTPUT_DIR "${output_dir}"

# rm -rf ${output_dir}
rm nohup.out

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-file configs/subset/cub.yaml \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "128" \
    MODEL.SUBSET.PERCENTILE "0.1" \
    MODEL.SUBSET.LN_GRAD "True" \
    MODEL.SUBSET.REINITIALIZE "False" \
    MODEL.SUBSET.REINITIALIZE_TYPE "constant" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    SOLVER.BASE_LR "0.01" \
    SOLVER.WEIGHT_DECAY "0.0001" \
    SEED "3407" \
    GPU_ID "${gpu_id}" \
    MODEL.MODEL_ROOT "${model_root}" \
    DATA.DATAPATH "${data_path}" \
    OUTPUT_DIR "${output_dir}"

