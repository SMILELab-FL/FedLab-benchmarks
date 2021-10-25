"""
Standalone mode for FedDyn, no server handler and no network manager.
Only client trainer is needed.
"""

import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

import argparse
import os
from pathlib import Path
import sys

sys.path.append("../../../FedLab/")

from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from fedlab.utils.functional import get_best_gpu, save_dict, load_dict

import models
from config import cifar10_config, balance_iid_data_config, debug_config
from client import FedDynSerialTrainer, FedDynSerialTrainer2, FedAvgSerialTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FedDyn-standalone demo in FedLab")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--client-num-per-rank", type=int, default=100)
    parser.add_argument("--alg", type=str, default='FedDyn')

    parser.add_argument("--partition", type=str, default='iid', help="Choose from ['iid', 'niid']")
    parser.add_argument("--data-dir", type=str, default='../../../datasets')
    parser.add_argument("--out-dir", type=str, default='./Output')
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # get basic config
    if args.partition == 'iid':
        alg_config = cifar10_config
        data_config = balance_iid_data_config
    # alg_config = debug_config

    # get basic model
    model = getattr(models, alg_config['model_name'])(alg_config['model_name'])

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if args.partition == 'iid':
        data_indices = load_dict(os.path.join('./Output', "cifar10_iid.pkl"))
    elif args.partition == 'noniid':
        data_indices = load_dict(os.path.join('./Output', "cifar10_noniid.pkl"))
    else:
        raise ValueError(f"args.partition '{args.partition}' is not supported yet")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'),
                                            train=True,
                                            download=False,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'),
                                           train=False,
                                           download=False,
                                           transform=transform_test)

    total_sample_num = len(trainset)

    weight_list = {
        cid: len(data_indices[cid]) / total_sample_num
        for cid in range(alg_config['num_clients'])
    }
    if args.alg == 'FedDyn':
        weight_list = {key: value * alg_config['num_clients'] for key, value in weight_list.items()}

    trainer_logger = Logger(f"StandaloneClientTrainer",
                            os.path.join(args.out_dir, f"ClientTrainer_Standalone.txt"))

    alg_config['out_dir'] = args.out_dir
    if args.alg == 'FedDyn':
        trainer = FedDynSerialTrainer2(model=model,
                                       dataset=trainset,
                                       data_slices=data_indices,
                                       client_weights=weight_list,
                                       rank=args.rank,
                                       logger=trainer_logger,
                                       args=alg_config)
    elif args.alg == 'FedAvg':
        trainer = FedAvgSerialTrainer(model=model,
                                      dataset=trainset,
                                      data_slices=data_indices,
                                      client_weights=weight_list,
                                      rank=args.rank,
                                      logger=trainer_logger,
                                      args=alg_config)
    else:
        raise ValueError(f"args.alg={args.alg} is not supported.")
