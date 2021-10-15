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


class FedDynSerialTrainer(SubsetSerialTrainer):
    def __init__(self, model,
                 dataset,
                 data_slices,
                 aggregator=None,
                 logger=Logger(),
                 cuda=True,
                 args=None):
        super().__init__(model,
                         dataset,
                         data_slices,
                         aggregator=None,
                         logger=logger,
                         cuda=cuda,
                         args=args)

    def _train_alone(self, model_parameters, train_loader):
        lr = self.args['lr']
        weight_decay = self.args['weight_decay']
        epochs = self.args['epochs']
        batch_size = self.args['batch_size']

        SerializationTool.deserialize_model(self._model, model_parameters)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.SGD(self._model.parameters(),
                                    lr=lr,
                                    weight_decay=alpha_coef + weight_decay)
        self._model.train()

        for e in range(epochs):
            # Training
            epoch_loss = 0
            for imgs, targets in train_loader:
                if self.cuda:
                    imgs, targets = imgs.cuda(self.gpu), targets.cuda(self.gpu)

                y_pred = self.model(imgs)

                # Get f_i estimate
                loss_f_i = loss_fn(y_pred, targets.reshape(-1).long())
                loss_f_i = loss_f_i / targets.shape[0]  # or    loss_f_i / list(batch_y.size())[0]
