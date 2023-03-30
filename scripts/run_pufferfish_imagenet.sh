#!/bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pufferfish_imagenet_training.py \
-a pufferfish_resnet50 \
--vanilla-arch resnet50 \
/workspace/ILSVRC2012/ \
--lr 0.1 \
--model-save-dir '/workspace/Cuttlefish' \
--lr-decay-period 30 60 80 \
--lr-decay-factor 0.1 \
--mode=lowrank \
--full-rank-warmup=True \
--re-warmup=True \
--fr-warmup-epoch=10 \
--lr-warmup= \
--warmup-epoch=5 \
-j 8 \
-p 100 \
--multiplier=16 \
-b 256



# #!/bin/bash

# cd ..

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python pufferfish_imagenet_training.py \
# -a pufferfish_wide_resnet50_2 \
# --vanilla-arch wide_resnet50_2 \
# /workspace/ILSVRC2012/ \
# --lr 0.1 \
# --model-save-dir '/workspace/Cuttlefish' \
# --lr-decay-period 30 60 80 \
# --lr-decay-factor 0.1 \
# --mode=lowrank \
# --full-rank-warmup=True \
# --re-warmup=True \
# --fr-warmup-epoch=6 \
# --lr-warmup= \
# --warmup-epoch=10 \
# -j 8 \
# -p 100 \
# --multiplier=16 \
# -b 256