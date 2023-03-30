export TASK_NAME=qnli
export EPOCH=20
export LR=2e-5
export num_gpus=2

python -m torch.distributed.launch --nproc_per_node=$num_gpus \
run_glue_roberta.py \
  --model_name_or_path bert-large-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate ${LR} \
  --num_train_epochs ${EPOCH} \
  --evaluation_strategy epoch \
  --save_steps 10000 \
  --overwrite_output_dir \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --output_dir /workspace/transformers/examples/pytorch/text-classification/results/${TASK_NAME}
