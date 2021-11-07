#!/bin/bash
# ============ Data Partition ============
# python data_partition.py --num-clients 100 --partition iid --balance True --dataset cifar10 --out-dir ./Output/ --seed 0
# echo -e "Data partition DONE.\n\n"

# ============ FedDyn ============
# SECONDS=0
# python standalone_main.py --num-clients 100 --sample-ratio 1.0 --alg FedDyn --out-dir ./Output/FedDyn/standalone
# FedDynELAPSED="Elapsed: $(($SECONDS / 60))min $(($SECONDS % 60))sec"
# echo $FedDynELAPSED

## ============ FedAvg ============
SECONDS=0
python standalone_main.py --num-clients 100 --sample-ratio 1.0 --alg FedAvg --out-dir ./Output/FedAvg/standalone-debug
FedAvgELAPSED="Elapsed: $(($SECONDS / 60))min $(($SECONDS % 60))sec"
echo $FedAvgELAPSED
