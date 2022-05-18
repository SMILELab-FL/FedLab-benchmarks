#!bin/bash

python server.py --ip 127.0.0.1 --port 3002 --world_size 6 --dataset mnist --round 100 &

python client.py --ip 127.0.0.1 --port 3002 --world_size 6 --rank 1 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 6 --rank 2 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 6 --rank 3 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 6 --rank 4 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 6 --rank 5 --dataset mnist &


wait