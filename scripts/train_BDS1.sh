#!/usr/bin/env bash
# bash ./scripts/train_dtu.sh  <experiment_name>
# 
# Ex: 
#   bash ./scripts/train_BDS1.sh BDS1_512x640_120s_N3_d192_250_2.5_itv1.36
#   bash ./scripts/train_BDS1.sh BDS1_512x640_120s_N3_d192_200_2.5_itv1.45
#   bash ./scripts/train_BDS1.sh BDS1_512x640_300s_N5_d192_200_2.5_itv1.25

TRAIN_PATH="/home/alcor/RECONSTRUCTION/EVAL_CODE/MVS/datasets/Blender/BDS1_mvs_training_512x640" \
TRAINLIST="lists/BDS1/train300.txt"
TESTLIST="lists/BDS1/test300.txt"
PAIRFILE="pair_33x10.txt"

# LOAD_CHKPT="./outputs/BDS1_512x640_120s_N3_d192_250_2.5_itv1.36/model_000127.ckpt"
LOAD_CHKPT="./outputs/BDS1_512x640_120s_N3_d192_200_2.5_itv1.45/model_000127.ckpt"

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
--dataset=blender  \
--trainpath=$TRAIN_PATH \
--logdir=$LOG_DIR \
--pairfile=$PAIRFILE \
--trainlist=$TRAINLIST \
--testlist=$TESTLIST \
--loadckpt=$LOAD_CHKPT \
--Nlights="1:1" \
--NtrainViews=5 \
--NtestViews=5 \
--numdepth=192  \
--interval_scale=1.25 \
--batch_size=3  \
--epochs=64 \
--lr=0.001 \
--lrepochs="1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30:1.2" \
--summary_freq=100 \
--seed=0 \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &

# --loadckpt=$LOAD_CHKPT \
