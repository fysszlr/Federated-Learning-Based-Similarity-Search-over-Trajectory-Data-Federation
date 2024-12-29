#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python train.py \
  --dataset porto \
  --cell_size 100 \
  --test_type distort \
  --method fcl