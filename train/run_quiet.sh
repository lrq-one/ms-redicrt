#!/usr/bin/env bash

ROOT="/home/lwh/projects/lrq2/fragnnet-main/ms2spectra_v1_r119"

cd "$ROOT"

mkdir -p runs

PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PYTHONPATH="$ROOT/code/src:$ROOT" \
python train/train.py all \
    2>&1 \
    | tee runs/mainline_from_scratch.raw.log \
    | PYTHONUNBUFFERED=1 \
      python train/_impl/quiet_progress.py

TRAIN_CODE=${PIPESTATUS[0]}

echo
echo "[TRAIN FINISHED] code=$TRAIN_CODE"
echo "[FULL LOG] runs/mainline_from_scratch.raw.log"
