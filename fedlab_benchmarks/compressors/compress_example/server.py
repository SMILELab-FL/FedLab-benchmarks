import argparse
import sys

from fedlab.core import communicator
from fedlab.utils.logger import Logger
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.compressor.topk import TopkCompressor
from fedlab.utils.message_code import MessageCode
from fedlab.core.communicator.processor import PackageProcessor

sys.path.append('../')
from models.cnn import CNN_MNIST

class CompressServerManager(ServerSynchronousManager):
    def __init__(self, network, handler, logger=None):
        super().__init__(network, handler, logger=logger)
        self.tpkc = TopkCompressor(compress_ratio=0.5)

    def on_receive(self, sender, message_code, payload):
        if message_code == MessageCode.ParameterUpdate:
            print(sender, message_code, payload[0].shape)
            #_, _, paylaod = PackageProcessor.recv_package(src=sender)
            #print("------", len(paylaod))
            return True
        else:
            raise Exception("Unexpected message code {}".format(message_code))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    model = CNN_MNIST()
    LOGGER = Logger(log_name="server")
    handler = SyncParameterServerHandler(model,
                                         global_round=args.round,
                                         logger=LOGGER,
                                         sample_ratio=args.sample)
    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0,
                          ethernet=args.ethernet)

    manager_ = CompressServerManager(handler=handler,
                                        network=network,
                                        logger=LOGGER)
    
    manager_.run()

