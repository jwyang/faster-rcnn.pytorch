#!/usr/bin/env bash

set -x
set -e

NET=$1
SESSION=$2
EPOCHS=$3
POINT=$4

DATASET="pascal_voc"

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="logs/test/${NET}_${DATASET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python test_net.py \
    --dataset ${DATASET} \
    --net ${NET} \
    --checksession ${SESSION} \
    --checkepoch ${EPOCHS} \
    --checkpoint ${POINT} \
    --cuda
