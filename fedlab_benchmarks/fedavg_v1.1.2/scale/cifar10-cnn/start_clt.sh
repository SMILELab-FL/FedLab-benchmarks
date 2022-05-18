#!/bin/bash
# TODO: try to add auto client assignment script
# TODO: bash start_clt.sh [world_size] [num_clients] [world_size]
# TODO: where world_size = 1 + client_ranks_num
# bash start_clt.sh  11 1 10 [iid/noniid]
for ((i=$2; i<=$3; i++))
do
{
    echo "client ${i} started"
    python client.py --world_size $1 --rank ${i} --setting $4&
    sleep 2s
}
done
wait
