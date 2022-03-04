from sys import path

path.append("../")
import torch
import argparse
import os

from torch import nn, optim
from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from ..client import FedProxTrainer
from setting import get_model, get_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str)

    parser.add_argument("--port", type=str)

    parser.add_argument("--world_size", type=int)

    parser.add_argument("--rank", type=int)

    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--epoch", type=int, default=5)

    parser.add_argument("--dataset", type=str)

    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--gpu", type=str, default="0,1,2,3")

    parser.add_argument("--ethernet", type=str, default=None)

    parser.add_argument(
        "--straggler", type=float, default=0.0
    )  # vaild value should be in range [0, 1] and mod 0.1 == 0

    parser.add_argument(
        "--optimizer", type=str, default="sgd"
    )  # valid value: {"sgd", "adam", "rmsprop"}

    parser.add_argument(
        "--mu", type=float, default=0.0
    )  # recommended value: {0.001, 0.01, 0.1, 1.0}

    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device(args.gpu)
    else:
        args.cuda = False
        device = torch.device("cpu")

    model = get_model(args).to(device)
    trainloader, testloader = get_dataset(args)
    optimizer_dict = dict(sgd=optim.SGD, adam=optim.Adam, rmsprop=optim.RMSprop)
    optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=args.rank,
        ethernet=args.ethernet,
    )

    LOGGER = Logger(log_name="client " + str(args.rank))

    trainer = FedProxTrainer(
        model=model,
        data_loader=trainloader,
        epochs=args.epoch,
        optimizer=optimizer,
        criterion=criterion,
        mu=args.mu,
        cuda=args.cuda,
        logger=LOGGER,
    )

    manager_ = ClientPassiveManager(trainer=trainer, network=network, logger=LOGGER)
    manager_.run()
