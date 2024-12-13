#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py --dataset beijing --cell_size 1000 --test_type distort --method fcl