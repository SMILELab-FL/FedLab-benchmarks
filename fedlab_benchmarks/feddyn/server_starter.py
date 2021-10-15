import argparse
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

import sys

sys.path.append("../../../FedLab/")

from fedlab.core.network import DistNetwork
from fedlab.core.server.scale.manager import ScaleSynchronousManager

import models
from config import cifar10_config, balance_iid_data_config
from server import RecodeHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3003")
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--ethernet', type=str, default=None)

    parser.add_argument("--partition", type=str, help="Choose from ['iid', 'niid']")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--data-dir", type=str, default='../../../datasets')
    parser.add_argument("--out-dir", type=str, default='./Output')
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # get basic model
    model = getattr(models, args.model_name)

    # get basic config
    # if args.partition == 'iid':
    alg_config = cifar10_config
    data_config = balance_iid_data_config

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=int(len(testset) / 10),
                                             drop_last=False,
                                             shuffle=False)

    handler = RecodeHandler(model,
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
