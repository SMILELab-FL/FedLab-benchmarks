#!/bin/bash

# Example: bash build_vocab.sh "../../datasets" "shakespeare" 0.25 30000 "./dataset_vocab"

dataset=$1
data_root=${2:-'../../datasets'}
data_select_ratio=${3:-'0.25'}
vocab_limit_size=${4:-'30000'}
save_root=${5:-'./dataset_vocab'}


python sample_build_vocab.py \
--dataset ${dataset} \
--data_root ${data_root} \
--data_select_ratio ${data_select_ratio} \
--vocab_limit_size ${vocab_limit_size} \
--save_root ${save_root}
