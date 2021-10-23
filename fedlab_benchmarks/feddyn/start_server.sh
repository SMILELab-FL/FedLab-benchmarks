#!/bin/bash
ClientRankNum=10
ClientNumPerRank=10
ClientNum=$(($ClientNumPerRank * $ClientRankNum))
WorldSize=$(($ClientRankNum + 1))
# balance iid cifar10 for 100 clients, check config.py for other setting
python data_partition.py --out-dir ./Output/ --partition iid --balance True --dataset cifar10 --num-clients ${ClientNum} --seed 0
echo -e "Data partition DONE.\n\n"
sleep 4s

python server_starter.py --world_size 11 --partition iid --alg FedAvg --out-dir ./Output/FedAvg/run1
