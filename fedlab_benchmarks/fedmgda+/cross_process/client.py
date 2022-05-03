from logging import log
import torch
import argparse
import sys
import os
import tqdm
from copy import deepcopy

import torchvision
from torchvision import transforms

from torch import nn
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.client.trainer import SGDClientTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils import Logger, SerializationTool
from fedlab.utils.functional import load_dict
from fedlab.utils.dataset import SubsetSampler

from setting import get_model, get_dataset


class ProxTrainer(SGDClientTrainer):
    """Refer to GitHub implementation https://github.com/WwZzz/easyFL """
    def __init__(
            self,
            model,
            data_loader,
            epochs,
            optimizer,
            criterion,
            mu,
            cuda=True,
            logger=Logger(),
    ):
        super().__init__(model,
                         data_loader,
                         epochs,
                         optimizer,
                         criterion,
                         cuda=cuda,
                         logger=logger)

        self.mu = mu
        self.delta_w = None

    @property
    def uplink_package(self):
        return self.delta_w

    def local_process(self, payload) -> None:
        model_parameters = payload[0]
        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, model_parameters)
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for inputs, labels in self._data_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(
                        self.gpu)

                outputs = self._model(inputs)
                l1 = self.criterion(outputs, labels)
                l2 = 0.0

                for w0, w in zip(frz_model.parameters(),
                                 self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * self.mu * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")
        self.delta_w = model_parameters - self.model_parameters

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str)
    parser.add_argument("--port", type=str)
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--mu", type=float, default=0.1)

    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str, default=None)
    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.cuda = False

    model = get_model(args)
    trainloader, testloader = get_dataset(args)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=args.rank,
        ethernet=args.ethernet,
    )

    LOGGER = Logger(log_name="client " + str(args.rank))

    trainer = ProxTrainer(model,
                          trainloader,
                          epochs=args.epoch,
                          optimizer=optimizer,
                          criterion=criterion,
                          mu=args.mu,
                          cuda=args.cuda,
                          logger=LOGGER)

    manager_ = PassiveClientManager(trainer=trainer,
                                    network=network,
                                    logger=LOGGER)
    manager_.run()
