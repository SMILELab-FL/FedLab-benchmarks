import argparse
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

torch.manual_seed(0)

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import evaluate

import sys

sys.path.append('../../../')
from models.rnn import RNN_Shakespeare
from leaf.dataloader import get_LEAF_all_test_dataloader

# python server.py --world_size 11
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=2)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=0.05)

    args = parser.parse_args()

    model = RNN_Shakespeare()

    handler = SyncParameterServerHandler(model,
                                         global_round=args.round,
                                         sample_ratio=args.sample,
                                         cuda=True)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler)
    manager_.run()
