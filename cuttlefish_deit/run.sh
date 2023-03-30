# DeiT-base
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=4 \
--use_env main.py \
--model-vanilla deit_base_patch16_224 \
--model adapt_deit_base_patch16_224 \
--factorized-mode adaptive \
--rank-est-metric=scaled-stable-rank \
--batch-size 256 \
--num_workers 6 \
--seed 4 \
--factorized-lr-decay 3.0 \
--data-path /workspace/ILSVRC2012/ \
--output_dir /workspace/Cuttlefish/cuttlefish_deit


# ResMLP
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
# --nproc_per_node=4 \
# --use_env main.py \
# --model-vanilla resmlp_36_224 \
# --model adapt_resmlp_36_224 \
# --factorized-mode adaptive \
# --rank-est-metric=scaled-stable-rank \
# --batch-size 256 \
# --num_workers 6 \
# --seed 4 \
# --factorized-lr-decay 3.0 \
# --data-path /workspace/ILSVRC2012/ \
# --output_dir /workspace/Cuttlefish/cuttlefish_deit