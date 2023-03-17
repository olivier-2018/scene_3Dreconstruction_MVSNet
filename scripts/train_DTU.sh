#!/usr/bin/env bash
# bash ./scripts/train_dtu.sh  <experiment_name>
# 
# Ex: 
#   bash ./scripts/train_DTU.sh  DTU_512x640_N3_d192_itv1.06

TRAIN_PATH="/home/alcor/RECONSTRUCTION/EVAL_CODE/MVS/datasets/DTU/mvs_training"
TRAINLIST="lists/dtu/train.txt"
TESTLIST="lists/dtu/test.txt"

exp=$1
PY_ARGS=${@:2}

LOG_DIR="./outputs/"$exp 
LOG_FILE="log_"$exp".txt"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

echo "====== Check log in file: tail -f ${LOG_DIR}/${LOG_FILE}"


python train.py \
--mode=train \
--dataset=dtu_yao  \
--trainpath=$TRAIN_PATH \
--logdir=$LOG_DIR \
--trainlist lists/dtu/train.txt  \
--testlist lists/dtu/test.txt  \
--numdepth=192  \
--interval_scale=1.06 \
--batch_size=4  \
--epochs=16 \
--lr=0.001 \
--lrepochs="10,12,14:2" \
--save_freq=10 \
--summary_freq=100 \
--seed=0 \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &

