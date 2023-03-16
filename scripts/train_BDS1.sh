#!/usr/bin/env bash
python train.py \
--mode train \
--model mvsnet \
--dataset=merlin  \
--trainpath="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Blender/BDS1_mvs_training_512x640"  \
--trainlist lists/merlin/train.txt  \
--testlist lists/merlin/test.txt  \
--epochs=33 \
--lr 0.001 \
--lrepochs="6,8,10,12,14,16,18,20,22,24,26,28,30:2" \
--batch_size=2  \
--numdepth=256  \
--interval_scale=2.5  \
--save_freq 1 \
--summary_freq 50  \
--logdir="./checkpoints/BSD1/d256_itvl2.5" \
--resume 
