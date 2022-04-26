#!bin/bash
python server.py --ip 127.0.0.1  --port 3002 --world_size 4  --rank 0 &

python client.py --ip 127.0.0.1  --port 3002 --world_size 4  --rank 1 &

python client.py --ip 127.0.0.1  --port 3002 --world_size 4  --rank 2 &

python client.py --ip 127.0.0.1  --port 3002 --world_size 4  --rank 3 &

wait