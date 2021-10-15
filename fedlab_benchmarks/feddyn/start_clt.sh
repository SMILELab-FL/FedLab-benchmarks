#!/bin/bash
# balance iid cifar10 for 100 clients, check config.py for other setting
python data_partition.py --partition iid --balance True --dataset cifar10 --num-clients 100 --seed 0
echo -e "Data partition DONE.\n\n"
sleep 2s

# launch client server
for ((i = 1; i <= 3; i++)); do
  {
    echo "client ${i} started"
    python client.py --world_size 4 --rank ${i} --client-num-per-rank 10 &
    sleep 2s
  }
done

wait
