import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

import argparse
import os
from pathlib import Path
import sys

sys.path.append("../../../FedLab/")

from fedlab.core.network import DistNetwork
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.utils.logger import Logger
from fedlab.utils.functional import load_dict

from config import cifar10_config, debug_config
from server import FedDynServerHandler, FedAvgServerHandler, FedAvgServerManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--ethernet', type=str, default=None)

    parser.add_argument("--partition", type=str, default='iid', help="Choose from ['iid', 'niid']")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--alg", type=str, default='FedDyn')
    parser.add_argument("--data-dir", type=str, default='../../../datasets')
    parser.add_argument("--out-dir", type=str, default='./Output')
    args = parser.parse_args()

    # ========== Get basic config ==========
    if args.debug is True:
        alg_config = debug_config
    else:
        if args.partition == 'iid':
            alg_config = cifar10_config

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    alg_config['out_dir'] = args.out_dir

    # ========== Get data partition and client weight based on sample number ==========
    if args.partition == 'iid':
        data_indices = load_dict(os.path.join(args.out_dir, "cifar10_iid.pkl"))
    else:
        raise ValueError(f"args.partition '{args.partition}' is not supported yet")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'),
                                           train=False,
                                           download=False,
                                           transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=alg_config['test_batch_size'],
                                             drop_last=False,
                                             shuffle=False)

    # ========== Build FL simulation ==========
    handler_logger = Logger("ServerHandler",
                            os.path.join(args.out_dir, "server_handler.txt"))

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)
    manager_logger = Logger("ServerManager", os.path.join(args.out_dir, "server_manager.txt"))

    if args.alg == 'FedDyn':
        handler = FedDynServerHandler(global_round=alg_config["round"],
                                      sample_ratio=alg_config["sample_ratio"],
                                      test_loader=testloader,
                                      cuda=True,
                                      logger=handler_logger,
                                      args=alg_config)
        manager = ScaleSynchronousManager(network=network, handler=handler, logger=manager_logger)

    elif args.alg == 'FedAvg':
        total_train_sample_num = sum([len(indices) for indices in data_indices.values()])
        weight_list = {cid: len(data_indices[cid]) / total_train_sample_num for cid in
                       range(alg_config['num_clients'])}

        handler = FedAvgServerHandler(global_round=alg_config["round"],
                                      sample_ratio=alg_config["sample_ratio"],
                                      test_loader=testloader,
                                      weight_list=weight_list,
                                      cuda=True,
                                      logger=handler_logger,
                                      args=alg_config)
        manager = FedAvgServerManager(network=network, handler=handler, logger=manager_logger)

    else:
        raise ValueError(f"args.alg={args.alg} is not supported.")

    manager.run()
