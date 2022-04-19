#!bin/bash

python server.py --ip 127.0.0.1 --port 3002 --world_size 11 --dataset mnist --round 3 &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 1 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 2 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 3 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 4 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 5 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 6 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 7 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 8 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 9 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 11 --rank 10 --dataset mnist &

wait