#!/bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cuttlefish_imagenet_training.py \
-a adapt_hybrid_resnet50 \
--vanilla-arch resnet50 \
/workspace/ILSVRC2012 \
--lr 0.1 \
--lr-decay-period 30 60 80 \
--lr-decay-factor 0.1 \
--mode=lowrank \
--full-rank-warmup=True \
--re-warmup=True \
--fr-warmup-epoch=91 \
--lr-warmup= \
--warmup-epoch=5 \
-j 8 \
-p 200 \
--multiplier=16 \
-b 256