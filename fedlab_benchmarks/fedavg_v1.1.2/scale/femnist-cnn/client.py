import torch
import argparse
import sys
import os

from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

from fedlab.core.client.scale.trainer import SerialTrainer
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict

sys.path.append("../../../")
from models.cnn import CNN_FEMNIST, AlexNet_CIFAR10, CNN_MNIST
from leaf.dataloader import get_LEAF_dataloader


class FEMNISTTrainer(SerialTrainer):
    def __init__(self,
                 model,
                 client_num,
                 aggregator,
                 cuda=True,
                 logger=None,
                 args=None):
        super().__init__(model,
                         client_num,
                         aggregator,
                         cuda=cuda,
                         logger=logger)
        self.args = args

    def _get_dataloader(self, client_id):
        trainloader, _ = get_LEAF_dataloader("femnist", client_id=client_id)
        return trainloader

    def _train_alone(self, model_parameters, train_loader):

        epochs, lr = self.args["epochs"], self.args["lr"]
        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._model.train()

        for _ in range(epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                output = self.model(data)

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3002")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)

    parser.add_argument("--partition", type=str, default="iid")
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str, default=None)

    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.cuda = False

    model = CNN_FEMNIST()

    aggregator = Aggregators.fedavg_aggregate

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)

    trainer = FEMNISTTrainer(model=model,
                             client_num=359,
                             aggregator=aggregator,
                             args={
                                 "batch_size": 100,
                                 "lr": 0.001,
                                 "epochs": 5
                             })

    manager_ = ScaleClientPassiveManager(trainer=trainer, network=network)

    manager_.run()
