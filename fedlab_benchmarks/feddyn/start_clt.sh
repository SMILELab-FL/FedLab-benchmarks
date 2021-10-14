#!/bin/bash
# balance iid cifar10 for 100 clients, check config.py for other setting
python data_partition.py --partition iid --balance True --dataset cifar10 --num-clients 100

# launch client server
# TODO

