#!/bin/bash

cd ..

# CUDA_VISIBLE_DEVICES=0 python imagenet_layer_benchmark.py \
# -a lowrank_resnet50_layer_benchmark \
# /workspace/ILSVRC2012/ \
# --lr 0.1 \
# --model-save-dir '/workspace/Cuttlefish' \
# --lr-decay-period 30 60 80 \
# --lr-decay-factor 0.1 \
# --rank-factor=0 \
# --lr-warmup= \
# -j 8 \
# -p 100 \
# --multiplier=16 \
# -b 128


CUDA_VISIBLE_DEVICES=0 python imagenet_layer_benchmark.py \
-a lowrank_deit_small_patch16_224 \
/workspace/ILSVRC2012/ \
--lr 0.1 \
--model-save-dir '/workspace/Cuttlefish' \
--lr-decay-period 30 60 80 \
--lr-decay-factor 0.1 \
--rank-factor=0 \
--lr-warmup= \
-j 8 \
-p 100 \
--multiplier=16 \
-b 128