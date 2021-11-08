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
from fedlab.utils.functional import save_dict, load_dict

import models
from config import cifar10_config, balance_iid_data_config, debug_config
from client import FedDynSerialTrainer, FedDynSerialTrainer_v2, FedAvgSerialTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedDyn client demo in FedLab")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3002")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--client-num-per-rank", type=int, default=10)
    parser.add_argument("--alg", type=str, default='FedDyn')
    parser.add_argument("--ethernet", type=str, default=None)

    parser.add_argument("--partition", type=str, default='iid', help="Choose from ['iid', 'niid']")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--data-dir", type=str, default='../../../datasets')
    parser.add_argument("--out-dir", type=str, default='./Output')
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # get basic config
    if args.debug is True:
        alg_config = debug_config
    else:
        if args.partition == 'iid':
            alg_config = cifar10_config

    # get basic model
    model = getattr(models, alg_config['model_name'])(alg_config['model_name'])

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if args.partition == 'iid':
        data_indices = load_dict(os.path.join(args.out_dir, "cifar10_iid.pkl"))
    elif args.partition == 'noniid':
        data_indices = load_dict(os.path.join(args.out_dir, "cifar10_noniid.pkl"))
    else:
        raise ValueError(f"args.partition '{args.partition}' is not supported yet")

    # Process rank x represent client id from (x-1) * client_num_per_rank - x * client_num_per_rank
    # e.g. rank 5 <--> client 40-50
    client_id_list = [
        i for i in
        range((args.rank - 1) * args.client_num_per_rank, args.rank * args.client_num_per_rank)
    ]

    # get corresponding data partition indices
    sub_data_indices = {
        idx: data_indices[cid]
        for idx, cid in enumerate(client_id_list)
    }

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'),
                                            train=True,
                                            download=False,
                                            transform=transform_train)

    total_sample_num = len(trainset)

    sub_client_weights = {
        idx: len(data_indices[cid]) / total_sample_num
        for idx, cid in enumerate(client_id_list)
    }
    if 'FedDyn' in args.alg:
        sub_client_weights = {key: value * alg_config['num_clients'] for key, value in
                              sub_client_weights.items()}

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)

    trainer_logger = Logger(f"ClientTrainer-Rank-{args.rank:02d}",
                            os.path.join(args.out_dir, f"ClientTrainer_rank_{args.rank:02d}.txt"))
    alg_config['out_dir'] = args.out_dir
    if args.alg == 'FedDyn':
        trainer = FedDynSerialTrainer(model=model,
                                      dataset=trainset,
                                      data_slices=sub_data_indices,
                                      transform=transform_train,
                                      client_weights=sub_client_weights,
                                      rank=args.rank,
                                      logger=trainer_logger,
                                      args=alg_config)
    elif args.alg=='FedDyn_v2':
        trainer = FedDynSerialTrainer_v2(model=model,
                                         dataset=trainset,
                                         data_slices=sub_data_indices,
                                         transform=transform_train,
                                         client_weights=sub_client_weights,
                                         rank=args.rank,
                                         logger=trainer_logger,
                                         args=alg_config)
    elif args.alg == 'FedAvg':
        trainer = FedAvgSerialTrainer(model=model,
                                      dataset=trainset,
                                      data_slices=sub_data_indices,
                                      transform=transform_train,
                                      client_weights=sub_client_weights,
                                      rank=args.rank,
                                      logger=trainer_logger,
                                      args=alg_config)
    else:
        raise ValueError(f"args.alg={args.alg} is not supported.")

    manager = ScaleClientPassiveManager(trainer=trainer, network=network)
    manager.run()
