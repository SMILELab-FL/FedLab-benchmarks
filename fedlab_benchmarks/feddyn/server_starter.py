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

from config import cifar10_config, balance_iid_data_config, debug_config
from server import FedDynServerHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--ethernet', type=str, default=None)

    parser.add_argument("--partition", type=str, default='iid', help="Choose from ['iid', 'niid']")
    parser.add_argument("--model-name", type=str)
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

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'),
                                            train=True,
                                            download=False,
                                            transform=transform_test)

    testset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'),
                                           train=False,
                                           download=False,
                                           transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=alg_config['test_batch_size'],
                                              drop_last=False,
                                              shuffle=False)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=alg_config['test_batch_size'],
                                             drop_last=False,
                                             shuffle=False)

    server_logger = Logger("ServerHandler",
                           os.path.join(args.out_dir, "server_handler.txt"))

    alg_config['out_dir'] = args.out_dir
    handler = FedDynServerHandler(global_round=alg_config["round"],
                                  sample_ratio=alg_config["sample_ratio"],
                                  test_loader=testloader,
                                  train_loader=trainloader,
                                  cuda=True,
                                  logger=server_logger,
                                  args=alg_config)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler, 
                                       logger=Logger("ServerManager",
                                                     os.path.join(args.out_dir, "server_manager.txt")))
    manager_.run()
