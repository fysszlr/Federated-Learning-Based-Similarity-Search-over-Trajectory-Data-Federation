#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py --dataset porto --cell_size 100 --test_type downsampling --method fedavg