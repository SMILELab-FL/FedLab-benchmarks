# FedAvg

FedAvg is the baseline of synchronous federated learning algorithm, and FedLab implements the algorithm flow of FedAvg, including standalone and Cross Process scenarios.

## Requirements

fedlab==1.1.2

## Standalone

The` SerialTrainer` module is for the FL system simulation on a single machine, and its source code can be found in `fedlab/core fedlab/core/client/trainer.py`.

Executable scripts is in ` fedlab_benchmarks/algorithm/fedavg/standalone/`.

## Cross Process

The federated simulation of **multi-machine** and **single-machine multi-process** scenarios is the core module of FedLab, which is composed of various modules in `core/client` and `core/server`, please refer to overview for details .

The executable script is in `fedlab_benchmarks/algorithm/fedavg/cross_process/`