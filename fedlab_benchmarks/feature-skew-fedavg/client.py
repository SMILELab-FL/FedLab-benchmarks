import argparse
import os
import logging
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

import sys

from fedlab.core.client import SERIAL_TRAINER
from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork

from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict
from fedlab.utils.dataset.sampler import SubsetSampler
from fedlab.core.communicator.processor import Package, PackageProcessor
from fedlab.core.coordinator import Coordinator
from fedlab.utils.functional import AverageMeter
from fedlab.utils.message_code import MessageCode

from models import SimpleCNNMNIST
from config import fmnist_noise_baseline_config


class Subset(torch.utils.data.Dataset):
    """For data subset with different augmentation.
    """
    def __init__(self, dataset, indices, transform=None):
        self.data, self.targets = [], []
        for idx in indices:
            img, label = dataset[idx]
            self.data.append(img)
            self.targets.append(label)

        self.transform = transform

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.targets)


class AddGaussianNoise(object):
    """
    This transform function is from NIID-bench official code:
    https://github.com/Xtra-Computing/NIID-Bench
    """

    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:, row * size + i, col * size + j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class FeatureSkewTrainer(SubsetSerialTrainer):
    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 aggregator=None,
                 logger=None,
                 cuda=True,
                 args=None) -> None:
        super(FeatureSkewTrainer, self).__init__(model,
                                                 dataset=dataset,
                                                 data_slices=data_slices,
                                                 aggregator=aggregator,
                                                 logger=logger,
                                                 cuda=cuda,
                                                 args=args)
        self.complete_dataset = dataset
        self.client_dataset = dict()

        num_clients = self.args['total_client_num']
        noise = self.args['noise']

        for cid in range(num_clients):
            if cid == num_clients - 1:
                noise_level = 0
            else:
                noise_level = noise / num_clients * (
                            cid + 1)  # a little different from original NIID-bench

            curr_client_dataset = Subset(dataset=self.complete_dataset,
                                         indices=self.data_slices[cid],
                                         transform=AddGaussianNoise(0., noise_level))
            self.client_dataset[cid] = curr_client_dataset

    def _get_dataloader(self, client_id):
        batch_size = self.args["batch_size"]
        train_loader = torch.utils.data.DataLoader(self.client_dataset[client_id],
                                                   batch_size=batch_size, shuffle=True)
        return train_loader

    def _train_alone(self, model_parameters, train_loader):
        epochs, lr = self.args["epochs"], self.args["lr"]
        momentum, weight_decay = self.args["momentum"], self.args["weight_decay"]
        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)  # use momentum & weight-decay here
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
    parser = argparse.ArgumentParser(description="FedAvg server example")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3003")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--client-num-per-rank", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--ethernet", type=str, default=None)

    parser.add_argument("--setting", type=str, default='noise')
    args = parser.parse_args()

    config = fmnist_noise_baseline_config

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if args.setting == 'noise':
        data_indices = load_dict("fmnist_iid.pkl")

    # Process rank x represent client id from (x-1) * client_num_per_rank - x * client_num_per_rank
    # e.g. rank 5 <--> client 40-50
    client_id_list = [
        i for i in
        range((args.rank - 1) * args.client_num_per_rank, args.rank * args.client_num_per_rank)
    ]

    # get corresponding data partition indices
    sub_data_indices = {
        idx: data_indices[cid]
        for idx, cid in enumerate(client_id_list)
    }

    model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)

    trainset = torchvision.datasets.FashionMNIST(root='../../../datasets/FMNIST/',
                                                 train=True,
                                                 download=False,
                                                 transform=transforms.ToTensor())

    aggregator = Aggregators.fedavg_aggregate

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)

    config['noise'] = args.noise  # add noise to configures
    trainer = FeatureSkewTrainer(model=model,
                                 dataset=trainset,
                                 data_slices=sub_data_indices,
                                 aggregator=aggregator,
                                 args=config)

    manager_ = ScaleClientPassiveManager(trainer=trainer, network=network)

    manager_.run()
