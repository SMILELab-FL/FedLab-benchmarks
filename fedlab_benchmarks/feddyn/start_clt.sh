#!/bin/bash
ClientRankNum=10
ClientNumPerRank=10
ClientNum=$(($ClientNumPerRank * $ClientRankNum))
WorldSize=$(($ClientRankNum + 1))
# balance iid cifar10 for 100 clients, check config.py for other setting
python data_partition.py --out-dir ./Output/run1 --partition iid --balance True --dataset cifar10 --num-clients ${ClientNum} --seed 0
echo -e "Data partition DONE.\n\n"
sleep 4s

# launch client server
for ((i = 1; i <= ${ClientRankNum}; i++)); do
  {
    echo "client ${i} started"
    python client_starter.py --world_size ${WorldSize} --rank ${i} --client-num-per-rank ${ClientNumPerRank} --out-dir ./Output/run1 &
    sleep 2s
  }
done

wait
