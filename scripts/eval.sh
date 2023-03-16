#!/usr/bin/env bash
DTU_TESTING="./DTU_TESTING"  # rectified test images folder 
CKPT_FILE="./checkpoints/d128/model_000009.ckpt"
python test.py --dataset=dtu_yao_eval --batch_size=1 --numdepth=128 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@
