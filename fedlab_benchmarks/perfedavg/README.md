# Personalized-FedAvg

paper

- Personalized-FedAvg: [Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/abs/1909.12488)
- Reptile: [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)
- MAML: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)


## Data

- Download train and test datasets manually or they will be automatically download to `fedlab_benchmarks/dataset/DATASET_NAME` from `torchvision.datasets`. You can redirect the root of datasets to your dataset directory, datasets root is set in `utils.get_datasets()`
- Experiments are run on EMNIST(default) and MNIST.

## Run


There're two way to run experiment in **Linux**. I have already set all hyper parameters well according to paper. Of course those can be modified. You can check `utils.get_args()` for more details about all hyper parameters. 


### Single-process

```python
python single_process.py
```

### Multi-process (needs more computational power)

I have set 2 workers(process) to handle all training tasks.

```python
cd multi_process/ ; sh quick_start.sh
```



## Performance

Evaluation result after fine-tuned is shown below. 

Communication round: `500`

Fine-tune: outer loop: `100`; inner loop: `10`

Personalization round: `5`

| FedAvg local training epochs (5 clients) | Initial loss | Initial Acc | Personalized loss | Personalized Acc |
| ---------------------------------------- | ------------ | ----------- | ----------------- | ---------------- |
| 20                                       | 2.6404       | 68.94%      | 0.2486            | **97.14%**       |
| 10                                       | 1.3143       | 71.20%      | 0.2716            | 93.20%           |
| 5                                        | 1.0677       | 72.27%      | 0.2288            | 94.16%           |
| 2                                        | 0.9335       | **75.89%**  | 0.2430            | 93.82%           |

| FedAvg local training epochs (20 clients) | Initial loss | Initial Acc | Personalized loss | Personalized Acc |
| ----------------------------------------- | ------------ | ----------- | ----------------- | ---------------- |
| 20                                        | 1.2058       | 81.64%      | 0.1067            | 98.73%           |
| 10                                        | 1.2833       | 81.61%      | 0.1007            | 98.72%           |
| 5                                         | 1.3430       | 78.35%      | 0.1055            | 98.79%           |
| 2                                         | 1.2059       | **82.63%**  | 0.0843            | **99.10%**       |

Experiment result from [paper](https://arxiv.org/abs/1909.12488) is shown below

![image-20220326202529457](image/image-20220326202529457.png)

