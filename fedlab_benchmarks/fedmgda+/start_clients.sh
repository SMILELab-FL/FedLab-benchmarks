#!/bin/bash

for ((i=$2; i<=$3; i++))
do
{
    echo "client ${i} started"
    python client.py --ip 127.0.0.1 --port 3002 --world_size $1 --rank ${i} --scale True &
    sleep 3s
}
done
wait