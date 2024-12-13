#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py --dataset beijing --cell_size 500 --test_type downsampling --method fcl