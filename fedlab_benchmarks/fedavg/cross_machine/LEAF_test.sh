#!/bin/bash

# get world size dicts for all datasets and run all client processes for each dataset
declare -A dataset_world_size=(['femnist']=5 ['shakespeare']=3) # each process represents one client/server

for key in ${!dataset_world_size[*]}
do
  echo "${key} client_num is ${dataset_world_size[${key}]}"

  echo "server started"
  python server.py --ip 127.0.0.1 --port 3002 --world_size ${dataset_world_size[${key}]}  --dataset ${key} &

  for ((i=1; i<${dataset_world_size[$key]}; i++))
  do
  {
      echo "client ${i} started"
      python client.py --ip 127.0.0.1 --port 3002 --world_size ${dataset_world_size[${key}]} --rank ${i} --dataset ${key} --epoch 2
  } &
  done
  wait

  echo "${key} experiment end"
done