#!/usr/bin/env bash
# bash ./scripts/train_dtu.sh  <experiment_name>
# 
# Ex: 
#   bash ./scripts/train_BDS8.sh BDS8_512x640_N5_d192_215_2.5_itv1.33_intrin548-548

TRAIN_PATH="./data/Blender/BDS8_mvs_train_512x640"
TRAINLIST="lists/BDS8/train200.txt"
TESTLIST="lists/BDS8/test200.txt"
PAIRFILE="pair_49x10.txt"

# LOAD_CHKPT="./outputs/BDS7_512x640_N5_d192_215_2.5_itv1.33_intrin548-548/model_000021.ckpt"

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
--Nlights="3:7" \
--NtrainViews=5 \
--NtestViews=5 \
--numdepth=192 \
--interval_scale=1.33 \
--batch_size=3 \
--epochs=24 \
--lr=0.000005 \
--lrepochs="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20:1.2" \
--summary_freq=100 \
--seed=0 \
--resume \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &

# --resume \
# --loadckpt=$LOAD_CHKPT \

