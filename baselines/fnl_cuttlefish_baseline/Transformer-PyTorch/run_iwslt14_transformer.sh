model=transformer
PROBLEM=IWSLT14_DEEN
SETTING=transformer_small


logdir=$model/$PROBLEM/$SETTING/alloutval0.25
mkdir -p $logdir

#CUDA_VISIBLE_DEVICES=0 python train.py ../data-bin/iwslt14.tokenized.de-en \
python train.py data-bin/iwslt14.tokenized.de-en \
	--arch $SETTING --save-dir $logdir \
	--clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--lr-scheduler inverse_sqrt --lr 0.25 --optimizer nag --warmup-init-lr 0.25 \
	--warmup-updates 4000 --max-update 100000  \
  --distributed-world-size 1 --device-id 0 --no-epoch-checkpoints \
  --weight-decay 1E-4 --rank-scale 0.25 \
  --spectral --spectral-quekey --spectral-outval \
  --wd2fd --wd2fd-outval

python generate.py data-bin/iwslt14.tokenized.de-en/ \
  --path $logdir/checkpoint_best.pt \
  --batch-size 128 --beam 5 --remove-bpe \
  --rank-scale 0.25 \
  --quiet --dump $logdir/bleu.log
