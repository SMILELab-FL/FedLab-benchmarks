## FedDyn

demo for FedDyn using FedLab in scale mode.

### Setting

- ``dataset``: CIFAR10
- `partition`: iid
- `balance`: `True`
- `batch_size`: 50
- ``num_clients``: 100
- `round`: 1000
- `epochs`: 5
- `lr`: 0.1
- `alpha_coef`: 1e-2
- `weight_decay`: 1e-3
- `max_norm`: 10
- `sample_ratio`: 1.0



### How to run?

`start_server.sh` is for server process launch, and `start_clt.sh` is for client process launch.

1. run command in terminal window 1 to launch server:

   ```bash
   bash start_server.sh
   ```

2. run command in terminal window 2 to launch clients:

   ```bash
   bash start_clt.sh
   ```

> random seed for data partiiton over clients can be set using `--seed` in `start_server.sh`:
>
> ```bash
> python data_partition.py --out-dir ./Output/FedDyn/run1 --partition iid --balance True --dataset cifar10 --num-clients ${ClientNum} --seed 1
> ```



<u>We highly recommend to launch clients after server is launched to avoid some conficts.</u>



### One-run Result

|                          | FedDyn-Paper | FedDyn-official | FedDyn-FedLab |
| ------------------------ | :----------: | :-------------: | :-----------: |
| Round for  $acc>81.40\%$ |      67      |       64        |      65       |
| Round for  $acc>85.00\%$ |     198      |       185       |      195      |

<img src="./Output/CIFAR10_100_iid_plots.png" height=400>

### Duration

|                |     FedAvg     |     FedDyn     |
| -------------- | :------------: | :------------: |
| FedDyn code    |   474.98 Min   |   537.13 Min   |
| FedLab (scale) | __160.90 Min__ | __253.17 Min__ |

### Environment

- CPU: 128G Memory, 32 cores, Intel(R) Core(TM) i9-9960X CPU @ 3.10GHz, 
- NVDIA GEFORCE RTX 2080 Ti



### Reference

- Acar, D. A. E., Zhao, Y., Matas, R., Mattina, M., Whatmough, P., & Saligrama, V. (2020, September). Federated learning based on dynamic regularization. In *International Conference on Learning Representations*.

