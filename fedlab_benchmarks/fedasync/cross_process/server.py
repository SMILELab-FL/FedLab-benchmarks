import os
import sys

from fedlab.core.network import DistNetwork
from fedlab.core.server.handler import AsyncParameterServerHandler
from fedlab.core.server.manager import ServerAsynchronousManager
sys.path.append("../../")
from models.cnn import CNN_MNIST
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    model = CNN_MNIST().cpu()
    ps = AsyncParameterServerHandler(model)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)
    Manager = ServerAsynchronousManager(handler=ps, network=network)

    Manager.run()
