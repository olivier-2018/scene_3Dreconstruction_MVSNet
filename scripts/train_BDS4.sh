#!/usr/bin/env bash
# bash ./scripts/train_dtu.sh  <experiment_name>
# 
# Ex: 
#   bash ./scripts/train_BDS4.sh BDS4_512x640_N5_d192_250_2.5_itv1.43

TRAIN_PATH="./data/Blender/BDS4_mvs_train_1024x1280"
TRAINLIST="lists/BDS4/train200.txt"
TESTLIST="lists/BDS4/test200.txt"
PAIRFILE="pair_40x10.txt"

# LOAD_CHKPT="./outputs/BDS1_512x640_N3_d192_250_2.5_itv1.36/model_000127.ckpt"
# LOAD_CHKPT="./outputs/BDS1_512x640_N3_d192_200_2.5_itv1.45/model_000127.ckpt"
# LOAD_CHKPT="./outputs/BDS1_512x640_300s_N5_d192_200_2.5_itv1.25/model_000050.ckpt"
# LOAD_CHKPT="./outputs/BDS2_512x640_N5_d192_300_2.5_itv1.31/model_000047.ckpt"


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
--Nlights=7 \
--NtrainViews=5 \
--NtestViews=5 \
--numdepth=192  \
--interval_scale=1.43 \
--batch_size=3  \
--epochs=24 \
--lr=0.001 \
--lrepochs="2,3,4,5,6,7,8,9,10,11,1213,14,15,16,17,18,20:1.2" \
--summary_freq=100 \
--seed=0 \
--resume \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &

# --resume \
# --loadckpt=$LOAD_CHKPT \

