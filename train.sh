#!/usr/bin/env bash

set -x
set -e

GPU_ID=$1
NET=$2
EPOCHS=$3

DATASET="pascal_voc"

LOG="logs/${NET}_${DATASET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=${GPU_ID} python trainval_net.py \
    --dataset pascal_voc \
    --net ${NET} \
    --epochs ${EPOCHS} \
    --cuda \
    --bs 1