#!/bin/bash

cd ..

SEED=0
TRIAL=0
EPOCHS=300
DATASET=cifar10
MODEL=resnet18

CUDA_VISIBLE_DEVICES=0 python main.py \
--arch=${MODEL} \
--mode=lowrank \
--rank-est-metric=scaled-stable-rank \
--dataset=${DATASET} \
--batch-size=1024 \
--epochs=${EPOCHS} \
--full-rank-warmup=True \
--fr-warmup-epoch=$((EPOCHS + 1)) \
--seed=${SEED} \
--lr=0.1 \
--frob-decay=True \
--extra-bns=False \
--resume=False \
--evaluate=False \
--scale-factor=8 \
--lr-warmup-epochs=5 \
--ckpt_path=./checkpoint/resnet18_best.pth \
--momentum=0.9


# SEED=0
# TRIAL=0
# EPOCHS=300
# DATASET=cifar10
# MODEL=resnet18
# WARMUP_EPOCH=80

# CUDA_VISIBLE_DEVICES=0 python main.py \
# --arch=${MODEL} \
# --mode=pufferfish \
# --rank-est-metric=scaled-stable-rank \
# --dataset=${DATASET} \
# --batch-size=1024 \
# --epochs=${EPOCHS} \
# --full-rank-warmup=True \
# --fr-warmup-epoch=${WARMUP_EPOCH} \
# --seed=${SEED} \
# --lr=0.1 \
# --frob-decay=False \
# --extra-bns=True \
# --resume=False \
# --evaluate=False \
# --scale-factor=8 \
# --lr-warmup-epochs=5 \
# --ckpt_path=./checkpoint/resnet18_best.pth \
# --momentum=0.9


# SEED=0
# TRIAL=0
# EPOCHS=200
# DATASET=svhn
# MODEL=vgg19

# CUDA_VISIBLE_DEVICES=0 python main.py \
# --arch=${MODEL} \
# --mode=lowrank \
# --rank-est-metric=scaled-stable-rank \
# --dataset=${DATASET} \
# --batch-size=1024 \
# --epochs=${EPOCHS} \
# --full-rank-warmup=True \
# --fr-warmup-epoch=$((EPOCHS + 1)) \
# --seed=${SEED} \
# --lr=0.1 \
# --frob-decay=False \
# --extra-bns=True \
# --resume=False \
# --evaluate=False \
# --scale-factor=8 \
# --lr-warmup-epochs=5 \
# --ckpt_path=./checkpoint/resnet18_best.pth \
# --momentum=0.9