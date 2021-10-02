#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train.py --train True --test False --batch_size 14 --max_len 512 --lr 5e-05 --epochs 10
CUDA_VISIBLE_DEVICES=1 python train.py --train False --test True --batch_size 14 --max_len 512
