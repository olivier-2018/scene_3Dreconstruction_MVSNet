#!/usr/bin/env bash
#   bash ./eval_Merlin.sh <weights_folder> <test_img_folder>
#

CKPT_FILE="checkpoints/BDS1/d256/model_000032.ckpt"
TESTPATH="data/Merlin/dataset-Mario_Set_with_GT_20220930/"  

run_folder=$1
experiment=$2
PY_ARGS=${@:3}

# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/dataset-Mario_Set_Blender_20221205"
# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/dataset-Mario_Set_Full_to_Empty_20220715"
TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/dataset-Mario_Set_with_GT_20220930"


# TESTLIST="lists/Merlin/test8.txt"
TESTLIST="lists/Merlin/test_scan5-7-10.txt"


python eval.py \
--dataset=merlin_eval \
--batch_size=1 \
--numdepth=128 \
--testpath=$TESTPATH \
--testlist lists/dtu/test.txt \
--loadckpt $CKPT_FILE $@
