## FedDyn Implementation using FedLab

先把可以通知communication round的demo实现：

- 方案1：用package传递
- 方案2：只在本地launch时候自动+1（这种比较简单）



如何打印client trainer的编号：直接获取rank（这样就比较脏）

### Setting

- ``n_clnt``: num_clients

