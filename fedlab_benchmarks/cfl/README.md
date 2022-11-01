# CFL

[Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/abs/1910.01991)


## Requirements

fedlab==1.2.1

## Run

We implement shifted and rotated data generation. And we accept the real-world data, e.g. femnist.

For shifted and rotated augmented data, we accept `mnist` and `cifar10` configuration now.

And we also provide the rotated 0 and 180 degree `emnist` used in the [original paper's code implementation](https://github.com/felisat/clustered-federated-learning#clustered-federated-learning-model-agnostic-distributed-multi-task-optimization-under-privacy-constraints)

If you don't have the augmented data, you should set the config `process_data=1` to generate data firstly. 
And the augmented data will be saved in `args.save_dir`, which defaults to `./datasets`.
You can see and modify the two parameters `save_dir` and `root`, which represent the augmented data storage path and origin data read path respectively.
`save_dir` defaults to `./datasets`, and `root` defaults to `../datasets/{dataset_name}`

For real-world data, we refer [LEAF](https://github.com/TalwalkarLab/leaf) to simulate. In our benchmark, we implement in `fedlab_benchmarks/leaf/`. 
You can read the docs in `leaf` folder to generate data. And cfl will use the defaulted leaf data saving path `../leaf/pickle_datasets/` to get data. 

## Performance

Null

## References

[1] Sattler, Felix, Klaus-Robert Müller, and Wojciech Samek. "Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints." arXiv preprint arXiv:1910.01991 (2019).
[2] [Resource Code](https://github.com/felisat/clustered-federated-learning#clustered-federated-learning-model-agnostic-distributed-multi-task-optimization-under-privacy-constraints)
