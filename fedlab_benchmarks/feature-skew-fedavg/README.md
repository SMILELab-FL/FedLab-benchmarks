# README

demo for NIID-bench "feature distribution skew"-"noise-based feature imbalance" on FedAvg.

__Setting:__
- `noise=0.1` (Gaussian noise)
- `partition='homo'` (IID partition)
- `n_parties=10`
- `lr=0.01`
- `momoentum=0.9`
- `weight_decay=1e-5` (L2-norm weight decay)
- `comm_round=50`
- `sample=1` (client sampling ratio)
- `alg='fedavg'`
- `epochs=10`
- `dataset='fmnist'`
- `model='simple-cnn'`

Top-1 accuracy for FMNIST in paper: $89.1\% \pm 0.3\%$.

Top-1 accuracy for FMNIST in this demo: $89.37\% \pm 0.14 \%$​​ (5 runs)

## Requirements

fedlab==1.1.2

## How to Run?

`start_server.sh` is for server process launch, and `start_clt.sh` is for client process launch.

1. run command in terminal window 1 to launch server:

   ```bash
   bash start_server.sh
   ```

2. run command in terminal window 2 to launch clients:

   ```bash
   bash start_client.sh
   ```

   > random seed for data partiiton over clients can be set using `--seed` in `start_clt.sh`:
   >
   > ```shell
   > python data_partition.py --num-clients 10 --seed 1
   > ```
   >
   > And the noise distribution can be send with `--noise`:
   >
   > ```bash
   > python client.py --world_size 2 --rank 1 --noise 0.1
   > ```



<u>We highly recommend to launch clients after server is launched to avoid some conficts.</u>



## References

- Li, Q., Diao, Y., Chen, Q., & He, B. (2021). Federated learning on non-iid data silos: An experimental study. *arXiv preprint arXiv:2102.02079*.

