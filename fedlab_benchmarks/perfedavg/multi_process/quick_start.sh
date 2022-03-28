#!bin/bash
if [ ! -d "../../dataset/emnist" ]; then
    mkdir ../../dataset/emnist ;
    python download_datasets.py ; 
fi

python server.py --ip 127.0.0.1  --port 3002 --world_size 3 --dataset emnist --rank 0 &

python client.py --ip 127.0.0.1  --port 3002 --world_size 3 --dataset emnist --rank 1 &

python client.py --ip 127.0.0.1  --port 3002 --world_size 3 --dataset emnist --rank 2 &

# python client.py --ip 127.0.0.1  --port 3002 --world_size 4 --dataset emnist --rank 3 &

wait