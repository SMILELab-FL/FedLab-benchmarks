#!/bin/bash


for ((i=$2; i<=$3; i++))
do
{
    echo "client ${i} started"
    python client.py --world_size $1 --rank ${i} --setting $4&
    sleep 2s
}
done
wait