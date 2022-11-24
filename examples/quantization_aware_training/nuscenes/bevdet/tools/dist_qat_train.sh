#!/usr/bin/env bash

CONFIG=$1
QCONFIG=$2
PRETRAINED=$3
GPUS=$4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/qat_train.py $CONFIG $QCONFIG $PRETRAINED --launcher pytorch ${@:5}
