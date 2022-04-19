import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms

import sys
sys.path.append('../../')

from fedlab.utils import Logger, SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.network import DistNetwork
from models.cnn import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST


class qfedavgServerHandler(SyncParameterServerHandler):

    def __init__(self,
                 model,
                 global_round,
                 sample_ratio,
                 cuda=False,
                 logger=None):
        super().__init__(model, global_round, sample_ratio, cuda, logger)

        self.local_losses = []
        self.lr = 0.01
        self.q = 5

        testset = torchvision.datasets.MNIST(root='../../datasets/mnist/',
                                             train=False,
                                             download=True,
                                             transform=transforms.ToTensor())

        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=int(
                                                          len(testset) / 10),
                                                      drop_last=False,
                                                      shuffle=False)

        self.model_list = []
        self.loss_list = []

    def _update_global_model(self, payload):
        model_parameters, loss = payload[0], payload[1].item()
        self.model_list.append(model_parameters)
        self.loss_list.append(loss)

        if len(self.model_list) == self.client_num_per_round:
            self.aggregation(self.model_list, self.loss_list)

            self.model_list = []
            self.loss_list = []

            self.round += 1
            return True

    def aggregation(self, model_parameters_list, client_loss):

        print("client losses: ", client_loss)

        gradients = [(self.model_parameters - parameters) / self.lr
                     for parameters in model_parameters_list]
        Deltas = [
            grad * np.float_power(loss + 1e-10, (self.q - 1))
            for grad, loss in zip(gradients, client_loss)
        ]
        hs = [
            self.q * np.float_power(loss + 1e-10,
                                    (self.q - 1)) * (grad.norm()**2) +
            1.0 / self.lr * np.float_power(loss + 1e-10, self.q)
            for grad, loss in zip(gradients, client_loss)
        ]

        demominator = np.sum(np.asarray(hs))
        scaled_deltas = [delta / demominator for delta in Deltas]
        updates = torch.Tensor(sum(scaled_deltas))

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

    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    model = CNN_MNIST()
    LOGGER = Logger(log_name="server")
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
