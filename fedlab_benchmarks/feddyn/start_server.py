import argparse
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

torch.manual_seed(0)

import models
from config import cifar10_config, balance_iid_data_config

import sys

sys.path.append("../../../FedLab/")
from fedlab.core.network import DistNetwork

# python server.py --world_size 11
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3003")
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--ethernet', type=str, default=None)

    parser.add_argument("--partition", type=str)
    args = parser.parse_args()

    if args.partition == 'iid':
        alg_config = cifar10_config
        data_config = balance_iid_data_config
    else:
        alg_config = None
        data_config = None

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='../../../../datasets/data/cifar10/',
        train=False,
        download=True,
        transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=int(len(testset) / 10),
                                             drop_last=False,
                                             shuffle=False)

    model = getattr(models, alg_config['model_name'])

    handler = RecodeHandler(model,
                            client_num_in_total=1,
                            global_round=alg_config["round"],
                            sample_ratio=alg_config["sample_ratio"],
                            test_loader=testloader,
                            cuda=True,
                            config=alg_config)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler)
    manager_.run()
