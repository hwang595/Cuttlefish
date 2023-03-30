model=transformer
PROBLEM=wmt14_enfr
#SETTING=transformer_base
SETTING=transformer_small

logdir=$model/$PROBLEM/$SETTING/frob
mkdir -p $logdir

python train.py data-bin/wmt14_en_fr \
	--clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
	--arch $SETTING --save-dir $logdir \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--lr-scheduler inverse_sqrt --lr 0.50 --optimizer nag --warmup-init-lr 0.50 \
	--warmup-updates 20000 --max-update 400000 --no-epoch-checkpoints \
  --device-id 0 --distributed-world-size 1 \
  --weight-decay 1E-4 --wd2fd-quekey --wd2fd-outval
#	--warmup-updates 16000 --max-update 300000 --save-interval 10000 \

python generate.py data-bin/wmt14_en_fr/ \
  --path $logdir/checkpoint_best.pt \
  --batch-size 128 --beam 5 --remove-bpe \
  --quiet --dump $logdir/bleu.log
