#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train_trajsimi.py --dataset beijing --trajsimi_measure_fn_name lcss --method fedavg