#!/usr/bin/env bash
# bash ./scripts/train_dtu.sh  <experiment_name>
# 
# Ex: 
#   bash ./scripts/train_BDS2.sh BDS2_512x640_N5_d192_300_2.5_itv1.31

TRAIN_PATH="/home/alcor/RECONSTRUCTION/EVAL_CODE/MVS/datasets/Blender/BDS2_mvs_train_512x640" \
TRAINLIST="lists/BDS2/train300.txt"
TESTLIST="lists/BDS2/test300.txt"
PAIRFILE="pair_48x10.txt"

# LOAD_CHKPT="./outputs/BDS1_512x640_N3_d192_250_2.5_itv1.36/model_000127.ckpt"
# LOAD_CHKPT="./outputs/BDS1_512x640_N3_d192_200_2.5_itv1.45/model_000127.ckpt"
LOAD_CHKPT="./outputs/BDS1_512x640_300s_N5_d192_200_2.5_itv1.25/model_000050.ckpt"


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
--dataset=blender \
--trainpath=$TRAIN_PATH \
--logdir=$LOG_DIR \
--pairfile=$PAIRFILE \
--trainlist=$TRAINLIST \
--testlist=$TESTLIST \
--Nlights="1:1" \
--NtrainViews=5 \
--NtestViews=5 \
--numdepth=192  \
--interval_scale=1.31 \
--batch_size=3  \
--epochs=64 \
--lr=0.001 \
--lrepochs="1,3,5,9,13:1.2" \
--summary_freq=100 \
--seed=0 \
--resume \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &

# --loadckpt=$LOAD_CHKPT \
