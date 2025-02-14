#! /bin/bash
export MASTER_ADDR=${1:-localhost}
export MASTER_PORT=${2:-10055}
export NODE_RANK=${3:-0}
export OMP_NUM_THREADS=6

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

echo $MASTER_ADDR
echo $MASTER_PORT

## Pretrain 262144
NODE_RANK=$NODE_RANK python main.py fit --config configs/IBQ/npu/pretrain_ibqgan_256_262144.yaml

## Pretrain 16384
# NODE_RANK=$NODE_RANK python main.py fit --config configs/IBQ/gpu/pretrain_ibqgan_256_16384.yaml

