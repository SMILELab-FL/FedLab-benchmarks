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

