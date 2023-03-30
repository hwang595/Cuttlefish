#!/bin/bash

cd ..

SEED=0
TRIAL=0
EPOCHS=300
DATASET=cifar10
MODEL=resnet18

CUDA_VISIBLE_DEVICES=0 python cifar_block_benchmark.py \
--arch=${MODEL} \
--dataset=${DATASET} \
--batch-size=1024 \
--epochs=${EPOCHS} \
--seed=${SEED} \
--lr=0.1 \
--rank-ratio=0.0 \
--resume=False \
--evaluate=False \
--scale-factor=8 \
--lr-warmup-epochs=5 \
--ckpt_path=./checkpoint/resnet18_best.pth \
--momentum=0.9