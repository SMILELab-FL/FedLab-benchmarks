from sys import path

path.append("../")
path.append("../../")

import argparse
from torch import nn
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from client_manager import PerFedAvgClientManager
from trainer import PerFedAvgTrainer
from fine_tuner import LocalFineTuner
from models import get_model
from utils import get_datasets, get_dataloader, get_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    datasets_root = (
        "../../datasets/emnist" if args.dataset == "emnist" else "../../datasets/mnist"
    )
    model = get_model(args)
    datasets = get_datasets(args, datasets_root)
    trainloader_list, valloader_list = get_dataloader(datasets, args)
    criterion = nn.CrossEntropyLoss()
    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=args.rank,
        ethernet=args.ethernet,
    )

    LOGGER = Logger(log_name="client process " + str(args.rank))

    perfedavg_trainer = PerFedAvgTrainer(
        model=model,
        trainloader_list=trainloader_list,
        valloader_list=valloader_list,
        optimizer_type="sgd",
        optimizer_args=dict(lr=args.local_lr),
        criterion=criterion,
        epochs=args.inner_loops,
        pers_round=args.pers_round,
        cuda=args.cuda,
        logger=Logger(log_name="node {}".format(args.rank)),
    )

    finetuner = LocalFineTuner(
        model=model,
        trainloader_list=trainloader_list,
        valloader_list=valloader_list,
        optimizer_type="adam",
        optimizer_args=dict(lr=args.fine_tune_local_lr, betas=(0, 0.999)),
        criterion=criterion,
        epochs=args.fine_tune_inner_loops,
        cuda=args.cuda,
        logger=Logger(log_name="node {}".format(args.rank)),
    )

    manager_ = PerFedAvgClientManager(
        network=network, fedavg_trainer=perfedavg_trainer, fine_tuner=finetuner,
    )
    manager_.run()
