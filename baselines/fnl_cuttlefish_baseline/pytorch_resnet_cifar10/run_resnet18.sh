#!/bin/bash
TRIAL=0
SEED=0
EPOCHS=300
DATASET=cifar10
RANKSCALE=0.08


CUDA_VISIBLE_DEVICES=0 python trainer.py \
--arch resnet18 --data ${DATASET} --rank-scale ${RANKSCALE} \
--batch-size 1024 --epochs ${EPOCHS} --scale-factor 8 \
--lr-warmup-epochs 5 --save-dir results/resnet20-factorized \
--spectral --wd2fd --seed=${SEED}