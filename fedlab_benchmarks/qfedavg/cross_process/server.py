import argparse

from fedlab.utils.logger import Logger
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork
from setting import get_model, get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    model = get_model(args)
    LOGGER = Logger(log_name="server")
    handler = SyncParameterServerHandler(model,
                                         global_round=args.round,
                                         logger=LOGGER,
                                         sample_ratio=args.sample)
                                         
    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0,
                          ethernet=args.ethernet)

    manager_ = ServerSynchronousManager(handler=handler,
                                        network=network,
                                        logger=LOGGER)
    manager_.run()
