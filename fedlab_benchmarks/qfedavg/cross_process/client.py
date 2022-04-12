from logging import log
import torch
import argparse
import sys
import os
import tqdm

from torch import nn
from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils import MessageCode, SerializationTool, Logger
from fedlab.core.communicator import PackageProcessor, Package
from setting import get_model, get_dataset



class qfedavgTrainer(ClientSGDTrainer):
    def train(self, model_parameters) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        print(type(self._data_loader))
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
        return self.model_parameters, ret_loss

class qfedavgManager(ClientPassiveManager):

    def main_loop(self):
        while True:
            sender_rank, message_code, payload = PackageProcessor.recv_package(src=0)
            if message_code == MessageCode.Exit:
                break
            elif message_code == MessageCode.ParameterUpdate:
                model_parameters = payload[0]
                model_parameters, train_loss = self._trainer.train(model_parameters=model_parameters)
                self.synchronize(torch.Tensor([train_loss]))
            else:
                raise ValueError("Invalid MessageCode {}. Please see MessageCode Enum".format(message_code))

    def synchronize(self, train_loss):
        self._LOGGER.info("synchronize model parameters with server")
        model_parameters = self._trainer.model_parameters
        pack = Package(message_code=MessageCode.ParameterUpdate,
                       content=[model_parameters, train_loss])
        PackageProcessor.send_package(pack, dst=0)

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

    trainer = qfedavgTrainer(
        model,
        trainloader,
        epochs=args.epoch,
        optimizer=optimizer,
        criterion=criterion,
        cuda=args.cuda,
        logger=LOGGER,
    )

    manager_ = qfedavgManager(trainer=trainer,
                                    network=network,
                                    logger=LOGGER) 
    manager_.run()
