DATASET=cifar10
TR=0.1
SEED=0
TRIAL=0

CUDA_VISIBLE_DEVICES=0 python main_pretrain.py --dataset ${DATASET} \
--network vgg \
--weight_decay 1E-4 \
--depth 19 \
--target-ratio ${TR} \
--batch_size 1024 \
--seed ${SEED} \
--scale-factor 8 \
--lr-warmup-epochs 5 \
--log_dir results/vgg19 \
--spectral \
--wd2fd