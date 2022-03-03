#!/bin/bash

# Example: bash gen_pickle_dataset.sh "shakespeare" "../datasets" "./pickle_datasets"
dataset=$1
data_root=${2:-'../datasets'}
pickle_root=${3:-'./pickle_datasets'}


python pickle_dataset.py \
--dataset ${dataset} \
--data_root ${data_root} \
--pickle_root ${pickle_root}
