export RES_DIR=/workspace/Cuttlefish/transformers/examples/pytorch/text-classification/results
export TASK_NAME=mrpc
export EPOCH=5
export LR=2e-5

rm -rf ${RES_DIR}/tinybert/${TASK_NAME}

CUDA_VISIBLE_DEVICES=0 python run_glue_vanilla.py \
  --model_name_or_path huawei-noah/TinyBERT_General_6L_768D \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate ${LR} \
  --num_train_epochs ${EPOCH} \
  --logging_steps 20 \
  --save_steps 10000 \
  --evaluation_strategy epoch \
  --overwrite_output_dir \
  --output_dir ${RES_DIR}/tinybert/${TASK_NAME}