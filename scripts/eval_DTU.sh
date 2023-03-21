#!/usr/bin/env bash
# bash ./scripts/eval_dtu.sh  <checkpoint_folder> <experiment_name>
# 
# Ex: 
#   bash ./scripts/eval_DTU.sh   DTU_512x640_N3_d192_itv1.06   DTU

chkpt_folder=$1
experiment=$2
PY_ARGS=${@:3}

TESTPATH="../datasets/DTU/mvs_training"
TESTLIST="lists/dtu/eval_only1.txt"
PAIRFILE="pair.txt"
CKPT_FILE="outputs/DTU_512x640_N3_d192_itv1.06/model_000017.ckpt"

NviewGen=2
NviewFilter=2
PHOTO_MASK=0.75
GEO_MASK=2
CONDMASK_PIX=1
CONDMASK_DEPTH=0.01
LIGHT_IDX=-3


run_experiment=$experiment"_N"$NVIEWS"_PhotoMask"$PHOTO_MASK"_GeoMask"$GEO_MASK"_GeoPix"$GEO_PIX"_GeoDepth"$GEO_DEPTH"_LightIdx"$LIGHT_IDX"_"$PAIRFILE
OUTDIR="./outputs/"$chkpt_folder"/"$run_experiment

if [ ! -d "$OUTDIR" ]; then
    echo "=== Creating log dir: "$OUTDIR
    mkdir -p $OUTDIR
fi
LOG_FILE="log_"$experiment".txt"
echo "=== Check log in file: tail -f  ${OUTDIR}/${LOG_FILE}"


python eval.py \
--dataset=dtu_yao_eval \
--testpath=$TESTPATH \
--outdir=$OUTDIR \
--testlist=$TESTLIST \
--pairfile=$PAIRFILE \
--batch_size=1 \
--numdepth=192 \
--interval_scale=1.06 \
--loadckpt=$CKPT_FILE \
--NviewGen=$NviewGen \
--NviewFilter=$NviewFilter \
--photomask=$PHOTO_MASK \
--geomask=$GEO_MASK \
--condmask_pixel=$CONDMASK_PIX \
--condmask_depth=$CONDMASK_DEPTH \
--debug_MVSnet=0 \
| tee -a $OUTDIR"/"$LOG_FILE &
