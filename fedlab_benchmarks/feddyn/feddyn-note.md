## FedDyn Implementation using FedLab

先把可以通知communication round的demo实现：

- 方案1：用package传递
- 方案2：只在本地launch时候自动+1（这种比较简单）





### Setting

- ``n_clnt``: num_clients





- machine09上在跑cifar10，simpleCNN，iid-balanced的实验
- 需要重新整理一下原论文给出的实验和参数



### Notes

FedAvg 1000 round:

- duration: 

  - 479.87 Min (machine14 official code)

  - 160.9 Min (machine14 FedLab)

    > 2021-10-23 23:54:49 ~ 2021-10-24 2:35:43

- start from 0.2 accuracy

- 

