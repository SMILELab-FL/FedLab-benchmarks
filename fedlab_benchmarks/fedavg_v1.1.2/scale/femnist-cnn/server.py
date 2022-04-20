import sys
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

torch.manual_seed(0)

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import AverageMeter

sys.path.append("../../../")
from models.cnn import CNN_FEMNIST

# python server.py --world_size 11
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=1000)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=0.01)

    args = parser.parse_args()

    model = CNN_FEMNIST()

    handler = SyncParameterServerHandler(model,
                                         global_round=args.round,
                                         sample_ratio=args.sample,
                                         cuda=True)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler)
    manager_.run()
