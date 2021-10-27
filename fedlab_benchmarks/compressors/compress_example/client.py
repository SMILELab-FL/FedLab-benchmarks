import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import argparse
import sys
import os

from fedlab.core.communicator import DATA_TYPE_FLOAT, DATA_TYPE_INT
from fedlab.core.communicator.package import Package
from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.core.network import DistNetwork
from fedlab.core.communicator.processor import PackageProcessor
from fedlab.core.communicator.package import Package
from fedlab.compressor.topk import TopkCompressor

from fedlab.utils.logger import Logger
from fedlab.utils.dataset.sampler import RawPartitionSampler
from fedlab.utils.message_code import MessageCode

sys.path.append('../')
from models.cnn import CNN_MNIST


class CompressClientManager(ClientPassiveManager):
    def __init__(self, network, trainer, logger=None):
        super().__init__(network, trainer, logger=logger)

        self.tpkc = TopkCompressor(compress_ratio=0.5)

    def synchronize(self):
        values_list, indices_list = self.tpkc.compress(
            self._trainer.model.parameters())

        print("send value")
        value_pack = Package(message_code=MessageCode.ParameterUpdate,
                             content=values_list)

        print(value_pack.content.shape)
        PackageProcessor.send_package(value_pack, dst=0)

        """
        
        print("send index")
        index_pack = Package(message_code=MessageCode.ParameterUpdate,
                             content=indices_list,
                             data_type=DATA_TYPE_INT)

        PackageProcessor.send_package(index_pack, dst=0)
        """

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

    model = CNN_MNIST()

    root = '../datasets/mnist/'
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root=root,
                                          train=True,
                                          download=True,
                                          transform=train_transform)

    testset = torchvision.datasets.MNIST(root=root,
                                         train=False,
                                         download=True,
                                         transform=test_transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=RawPartitionSampler(trainset,
                                    client_id=args.rank,
                                    num_replicas=args.world_size - 1),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.world_size)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=int(len(testset) / 10),
                                             drop_last=False,
                                             shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)

    LOGGER = Logger(log_name="client " + str(args.rank))

    trainer = ClientSGDTrainer(model,
                               trainloader,
                               epochs=args.epoch,
                               optimizer=optimizer,
                               criterion=criterion,
                               cuda=args.cuda,
                               logger=LOGGER)

    manager_ = CompressClientManager(trainer=trainer,
                                     network=network,
                                     logger=LOGGER)
    manager_.run()
