#!/bin/bash
# ==========================================
# =============== EXPERIMENT ===============
# ==========================================
ClientRankNum=10
ClientNumPerRank=10
ClientNum=$(($ClientNumPerRank * $ClientRankNum))
WorldSize=$(($ClientRankNum + 1))

# ------ FedAvg
# for ((i = 1; i <= ${ClientRankNum}; i++)); do
#   {
#     echo "client ${i} started"
#     python client_starter.py --world_size ${WorldSize} --rank ${i} --client-num-per-rank ${ClientNumPerRank} --alg FedAvg --out-dir ./Output/FedAvg/run1 &
#     sleep 2s
#   }
# done

# wait

# ------ FedDyn
for ((i = 1; i <= ${ClientRankNum}; i++)); do
 {
   echo "client ${i} started"
   python client_starter.py --world_size ${WorldSize} --rank ${i} --client-num-per-rank ${ClientNumPerRank} --alg FedDyn --out-dir ./Output/FedDyn/run1 &
   sleep 2s
 }
done

wait


