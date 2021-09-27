# -*- coding: utf-8 -*-
# @Time    : 9/27/21 12:48 AM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : client.py
# @Software: PyCharm
import torch
import argparse
import os

from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

import models
from config import cifar10_config, balance_iid_data_config, balance_iid_data_config

import sys

sys.path.append("../../../FedLab/")
from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork

from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedDyn implementation: Client scale mode")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3003")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--num-client-per-rank", type=int, default=10)
    parser.add_argument("--ethernet", type=str, default=None)

    parser.add_argument("--setting", type=str)
    args = parser.parse_args()

    if args.setting == 'iid':
        alg_config = cifar10_config
        data_config = balance_iid_data_config
    else:
        config = None

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='../../../datasets/cifar10/',
        train=True,
        download=True,
        transform=transform_train)

    if data_config['partition'] == "noniid":
        data_indices = load_dict("cifar10_noniid.pkl")
    if data_config['partition'] == "iid":
        data_indices = load_dict("cifar10_iid.pkl")

    # Process rank x represent client id from (x-1)*10 - (x-1)*10 +10
    # e.g. rank 5 <--> client 40-50
    client_id_list = [
        i for i in
        range((args.rank - 1) * args.num_client_per_rank, args.rank * args.num_client_per_rank)
    ]

    # get corresponding data partition indices
    sub_data_indices = {
        idx: data_indices[cid]
        for idx, cid in enumerate(client_id_list)
    }

    model = AlexNet_CIFAR10()

    aggregator = Aggregators.fedavg_aggregate

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)

    trainer = SubsetSerialTrainer(model=model,
                                  dataset=trainset,
                                  data_slices=sub_data_indices,
                                  aggregator=aggregator,
                                  args=config)

    manager_ = ScaleClientPassiveManager(trainer=trainer, network=network)

    manager_.run()
