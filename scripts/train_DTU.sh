#!/usr/bin/env bash
python train.py \
--mode train \
--model mvsnet \
--dataset=dtu_yao  \
--trainpath="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/DTU/mvs_training_512x640"  \
--trainlist lists/dtu/train.txt  \
--testlist lists/dtu/test.txt  \
--epochs=32 \
--lr 0.001 \
--lrepochs="10,12,14,16,18,20,22,24,26,28,30:2" \
--batch_size=2  \
--numdepth=128  \
--interval_scale=1.6 \
--save_freq 1 \
--summary_freq 50 \
--logdir="./checkpoints/DTU/d128_itvl4" \
--resume