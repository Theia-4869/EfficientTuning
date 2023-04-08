#!/bin/bash

for dataset in "caltech101" "dtd" "oxford_flowers102" "oxford_iiit_pet" "svhn_cropped" "sun397" "patch_camelyon" "eurosat" "resisc45" "clevr" "dmlab" "kitti" "dsprites" "smallnorb"; do
     bypy -v upload /data/shz/tuning_project/datasets/VTAB/${dataset}.zip dataset/VTAB/
done
