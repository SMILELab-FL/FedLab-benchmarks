## FedDyn Implementation using FedLab

### Setting

- ``n_clnt``: num_clients



- machine09上在跑cifar10，simpleCNN，iid-balanced的实验
- 需要重新整理一下原论文给出的实验和参数
- machine14上串行FedAvg很慢（14h还没跑完500 round），试图在machine13上改进`get_data_loader`，提前slice subdataset并存在serialtrainer里



### Notes

FedAvg 1000 round:

- duration: 
  - machine14 official code
    - run1: 479.87 Min 
    - run2: 474.98 Min
  - machine14 FedLab
    - Scale mode: 160.9 Min
    - Standalone: 919 Min (dataloader的问题，太慢了)



FedDyn 1000 Round:

- duration:

  - machine14 official code:
    - run1: 537.13 Min
    - 

  - 288.8 Min (machine14 FedLab with local file save)

