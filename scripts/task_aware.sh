#!/bin/bash

#MY_PYTHON="CUDA_VISIBLE_DEVICES=4 python"
gpu=$1
path=results/

CIFAR_100i="--save_path $path --batch_size 10 --cuda yes --seed 0 --n_epochs 1 --inner_steps 2 --replay_batch_size 64"

#CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model mufan --data_file mini --n_tasks 20 --n_runs 5 --n_memories 50 --train_csv data/mini_cl_train.csv --test_csv data/mini_cl_test.csv
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model mufan --data_file core --n_tasks 10 --n_runs 5 --n_memories 50 --train_csv data/core50_tr.csv --test_csv data/core50_te.csv --batch_size 32 --replay_batch_size 32



