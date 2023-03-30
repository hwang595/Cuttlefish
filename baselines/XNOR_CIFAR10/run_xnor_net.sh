#!/bin/bash

SEED=0
TRIAL=0
EPOCHS=300
DATASET=cifar10
MODEL=resnet18

CUDA_VISIBLE_DEVICES=0 python main.py \
--arch=${MODEL} \
--dataset=${DATASET} \
--batch-size=1024 \
--epochs=${EPOCHS} \
--seed=${SEED} \
--lr=0.001 \
--scale-factor=8 \
--lr-warmup-epochs=5 \
--momentum=0.9