from logging import log
import torch
import argparse
import sys
sys.path.append('../../')
import os
import tqdm
import torchvision
import torchvision.transforms as transforms

from torch import nn
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.client.trainer import SGDClientTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils import MessageCode, SerializationTool, Logger
from fedlab.utils.functional import load_dict
from fedlab.utils.dataset import SubsetSampler


from models.cnn import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST

class qfedavgTrainer(SGDClientTrainer):

    def __init__(self, model, data_loader, epochs, optimizer, criterion, cuda=False, logger=None):
        super().__init__(model, data_loader, epochs, optimizer, criterion, cuda, logger)

        self.loss = 0.0

    @property
    def uplink_package(self):
        return [self.model_parameters, torch.Tensor([self.loss])]

    def local_process(self, payload):
        model_parameters = payload[0]
        self.train(model_parameters)

    def train(self, model_parameters) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            ret_loss = 0.0
            for inputs, labels in self._data_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(
                        self.gpu)

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            ret_loss += loss.detach().item()
        self._LOGGER.info("Local train procedure is finished")
        self.loss = ret_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str)
    parser.add_argument("--port", type=str)
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str, default=None)
    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.cuda = False

    root = '../../datasets/mnist/'
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_indices = load_dict("mnist_noniid_200_10.pkl")

    trainset = torchvision.datasets.MNIST(root=root,
                                            train=True,
                                            download=True,
                                            transform=train_transform)

    trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=SubsetSampler(indices=data_indices[args.rank],
                                  shuffle=True),
            batch_size=args.batch_size)

    model = CNN_MNIST()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=args.rank,
        ethernet=args.ethernet,
    )

    LOGGER = Logger(log_name="client " + str(args.rank))

    trainer = qfedavgTrainer(
        model,
        trainloader,
        epochs=args.epoch,
        optimizer=optimizer,
        criterion=criterion,
        cuda=args.cuda,
        logger=LOGGER,
    )

    manager_ = PassiveClientManager(trainer=trainer,
                                    network=network,
                                    logger=LOGGER) 
    manager_.run()
