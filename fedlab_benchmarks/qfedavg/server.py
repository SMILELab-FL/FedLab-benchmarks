import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms

import sys
sys.path.append('../')

from fedlab.utils import Logger, SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.network import DistNetwork
from models.cnn import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST
from fedlab.utils import Aggregators

class qfedavgServerHandler(SyncParameterServerHandler):

    def __init__(self,
                 model,
                 global_round,
                 sample_ratio,
                 cuda=False,
                 logger=None):
        super().__init__(model, global_round, sample_ratio, cuda, logger)

        self.local_losses = []

        testset = torchvision.datasets.MNIST(root='../datasets/mnist/',
                                             train=False,
                                             download=True,
                                             transform=transforms.ToTensor())

        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=int(
                                                          len(testset) / 10),
                                                      drop_last=False,
                                                      shuffle=False)

        self.deltas = []
        self.hks = []
        self.client_loss = []
        self.parameters = []

    def _update_global_model(self, payload):
        
        self.deltas.append(payload[0])
        self.hks.append(payload[1])
        self.client_loss.append(payload[2].item())
        
        if len(self.deltas) == self.client_num_per_round:
            self.aggregation()

            self.deltas = []
            self.hks = []
            self.client_loss = []
            self.parameters = []

            self.round += 1
            return True

    def aggregation(self):

        hk = sum(self.hks)
        updates = sum([delta/hk for delta in self.deltas])
        model_parameters = self.model_parameters - updates

        SerializationTool.deserialize_model(self._model, model_parameters)
        loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                             self.testloader)
        self._LOGGER.info(
            "check check Server evaluate loss {:.4f}, acc {:.4f}".format(
                loss, acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=20)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    model = CNN_MNIST()
    LOGGER = Logger(log_name="server", log_file="./test.txt")
    handler = qfedavgServerHandler(model,
                                   global_round=args.round,
                                   logger=LOGGER,
                                   sample_ratio=args.sample)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0,
                          ethernet=args.ethernet)

    manager_ = SynchronousServerManager(handler=handler,
                                        network=network,
                                        logger=LOGGER)
    manager_.run()
