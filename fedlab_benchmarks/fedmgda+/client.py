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
from fedlab.core.client.serial_trainer import SubsetSerialTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils import Logger, SerializationTool
from fedlab.utils.functional import load_dict
from fedlab.utils.dataset import SubsetSampler

from setting import get_model, get_dataloader


class SerialProxTrainer(SubsetSerialTrainer):
    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 optimizer,
                 criterion,
                 logger=None,
                 cuda=False,
                 args=None) -> None:
        super().__init__(model, dataset, data_slices, logger, cuda, args)
        self.optimizer = optimizer
        self.criterion = criterion

    @property
    def uplink_package(self):
        return super().uplink_package

    def _get_dataloader(self, client_id):
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[client_id],
                                  shuffle=True),
            batch_size=self.args.batch_size)
        return train_loader

    def _train_alone(self, model_parameters, train_loader):
        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, model_parameters)
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.args.epochs):
            self._model.train()
            for inputs, labels in train_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(
                        self.gpu)

                outputs = self._model(inputs)
                l1 = self.criterion(outputs, labels)
                l2 = 0.0

                for w0, w in zip(frz_model.parameters(),
                                 self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * self.args.mu * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")
        return model_parameters - self.model_parameters


class ProxTrainer(SGDClientTrainer):
    """Refer to GitHub implementation https://github.com/WwZzz/easyFL """
    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=True,
                 logger=Logger(),
                 args=None):
        super().__init__(model,
                         data_loader,
                         epochs,
                         optimizer,
                         criterion,
                         cuda=cuda,
                         logger=logger)
        self.delta_w = None
        self.args = args

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

                loss = l1 + 0.5 * self.args.mu * l2

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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--mu", type=float, default=0.1)

    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str, default=None)
    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.cuda = False

    model = get_model(args)

    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=args.rank,
        ethernet=args.ethernet,
    )

    LOGGER = Logger(log_name="client " + str(args.rank))

    if not args.scale:
        trainloader, _ = get_dataloader(args)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        trainer = ProxTrainer(model,
                              trainloader,
                              epochs=args.epochs,
                              optimizer=optimizer,
                              criterion=criterion,
                              cuda=args.cuda,
                              logger=LOGGER,
                              args=args)
    else:
        data_slices = load_dict("mnist_noniid_200_100.pkl")
        client_id_list = [
            i for i in range((args.rank - 1) * 10, (args.rank - 1) * 10 + 10)
        ]
        # get corresponding data partition indices
        sub_data_indices = {
            idx: data_slices[cid]
            for idx, cid in enumerate(client_id_list)
        }
        root = '../datasets/mnist/'
        trainset = torchvision.datasets.MNIST(root=root,
                                              train=True,
                                              download=True,
                                              transform=transforms.ToTensor())
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        trainer = SerialProxTrainer(model,
                                    trainset,
                                    data_slices=sub_data_indices,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    cuda=args.cuda,
                                    logger=LOGGER,
                                    args=args)

    manager_ = PassiveClientManager(trainer=trainer,
                                    network=network,
                                    logger=LOGGER)
    manager_.run()
