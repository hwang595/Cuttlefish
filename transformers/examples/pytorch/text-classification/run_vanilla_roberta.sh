export TASK_NAME=qqp

CUDA_VISIBLE_DEVICES=0,1 python run_glue_vanilla_roberta.py \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --save_steps 50000 \
  --overwrite_output_dir \
  --output_dir /workspace/hongyi/transformers/examples/pytorch/text-classification/results/${TASK_NAME} > vanilla_roberta_bsize16_${TASK_NAME}_lr2e-5_trial0 2>&1
