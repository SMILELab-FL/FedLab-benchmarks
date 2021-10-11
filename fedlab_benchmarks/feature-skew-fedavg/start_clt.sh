#!/bin/bash
# perform data partition over clients
python data_partition.py --num-clients 10 --seed 1
echo -e "New data partition done \n\n"
# launch 10 clients in single serial trainer
python client.py --world_size 2 --rank 1 --noise 0.1
