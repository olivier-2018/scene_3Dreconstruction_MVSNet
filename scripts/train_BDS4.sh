#!/usr/bin/env bash
python train.py \
--mode train \
--model mvsnet \
--dataset=merlin  \
--trainpath="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Blender/BDS2_mvs_training_512x640"  \
--trainlist lists/blender/train.txt  \
--testlist lists/blender/test.txt  \
--epochs=26 \
--lr 0.001 \
--lrepochs="10,12,14,16,18,20,22,24:2" \
--batch_size=5  \
--numdepth=128  \
--interval_scale=2.2  \
--save_freq 1 \
--summary_freq 50  \
--logdir="./checkpoints/BDS2/d128_itvl5.5_4views" 
