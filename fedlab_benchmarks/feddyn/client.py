# -*- coding: utf-8 -*-
# @Time    : 9/27/21 12:48 AM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : client.py
# @Software: PyCharm

import argparse
import os

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

import models
from config import cifar10_config, balance_iid_data_config

import sys

sys.path.append("../../../FedLab/")
from fedlab.core.client import SERIAL_TRAINER, ORDINARY_TRAINER
from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.network import DistNetwork
from fedlab.core.communicator import Package, PackageProcessor

from fedlab.utils import MessageCode, SerializationTool, Aggregators
from fedlab.utils.logger import Logger
from fedlab.utils.functional import load_dict
from fedlab.utils.dataset import functional as dataF


class FedDynScaleClientPassiveManager(ClientPassiveManager):
    """Special client manager for :class:`SerialTrainer`.

    We modify the communication agreements creating mapping between process rank and client id.
    In this way, :class:`Manager` is able to represent multiple clients.
    Args:
        network (DistNetwork): Distributed network to use.
        handler (ClientTrainer): Subclass of :class:`ClientTrainer`, providing :meth:`train` and :attr:`model`.
    """

    def __init__(self, network, trainer):
        super().__init__(network, trainer)
        # only after receiving first package from server,
        # curr_round will begin to update
        self.curr_round = -1

    def on_receive(self, sender_rank, message_code, payload):
        """Actions to perform when receiving new message, including local training
        .. note::
            Customize the control flow of client corresponding with :class:`MessageCode`.
        Args:
            sender_rank (int): Rank of sender
            message_code (MessageCode): Agreements code defined in :class:`MessageCode`
            payload (list[torch.Tensor]): A list of tensors received from sender.
        """
        if message_code == MessageCode.ParameterUpdate:
            self.curr_round = int(payload[0])
            model_parameters = payload[1]

            _, message_code, payload = PackageProcessor.recv_package(src=0)
            id_list = payload[0].tolist()

            # check the trainer type
            if self._trainer.type == SERIAL_TRAINER:
                self.model_parameters_list = self._trainer.train(
                    model_parameters=model_parameters,
                    id_list=id_list,
                    curr_round=self.curr_round,
                    aggregate=False)
            elif self._trainer.type == ORDINARY_TRAINER:
                self.model_parameters_list = self._trainer.train(
                    model_parameters=model_parameters)

    def synchronize(self):
        """Synchronize local model with server actively
        .. note::
            Communication agreements related. Overwrite this function to customize package for synchronizing.
        """
        pack = Package(message_code=MessageCode.ParameterUpdate,
                       content=self.model_parameters_list)
        PackageProcessor.send_package(package=pack, dst=0)


class FedDynSerialTrainer(SubsetSerialTrainer):
    # no need to rewrite __init__
    def __init__(self, model, dataset, data_slices,
                 aggregator=None,
                 logger=None,
                 global_weight_list=None,
                 sub_weight_list=None,
                 cuda=True,
                 args=None):
        super(FedDynSerialTrainer, self).__init__(model=model,
                                                  dataset=dataset,
                                                  data_slices=data_slices,
                                                  aggregator=aggregator,
                                                  logger=logger,
                                                  cuda=cuda,
                                                  args=args)
        self.global_weight_list = global_weight_list
        self.sub_weight_list = sub_weight_list
        self.sub_alpha_coef_adpt = args['alpha_coef'] / sub_weight_list
        self.curr_round = -1

    def _train_alone(self, model_parameters,
                     client_id,
                     train_loader,
                     client_sample_num,
                     lr,
                     alpha_coef,
                     weight_decay):
        """Single round of local training for one client.
        Note:
            Overwrite this method to customize the PyTorch training pipeline.
        Args:
            model_parameters (torch.Tensor): model parameters.
            train_loader (torch.utils.data.DataLoader): dataloader for data iteration.
            client_sample_num (int): sample number for current client
        """
        args = self.args
        epochs, lr = args['epochs'], args['lr']
        SerializationTool.deserialize_model(self._model, model_parameters)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')  # loss function

        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr,
                                    weight_decay=alpha_coef + weight_decay)
        self._model.train()

        for epoch in range(epochs):
            # Training
            epoch_loss = 0
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                y_pred = self._model(data)

                ## Get f_i estimate
                loss_f_i = loss_fn(y_pred, target.reshape(-1).long())
                loss_f_i = loss_f_i / list(target.size())[0]

                # Get linear penalty on the current parameter estimates
                local_par_list = None
                for param in self._model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = args['alpha_coef'] * torch.sum(
                    local_par_list * (- avg_mdl_param + local_grad_vector))
                loss = loss_f_i + loss_algo

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                               max_norm=args['max_norm'])  # Clip gradients
                optimizer.step()
                epoch_loss += loss.item() * list(target.size())[0]

            if (epoch + 1) % args['print_per'] == 0:
                epoch_loss /= client_sample_num
                if args['weight_decay'] is not None:
                    # Add L2 loss to complete f_i
                    params = self.model_parameters
                    epoch_loss += (alpha_coef + weight_decay) / 2 * torch.sum(
                        params * params)
                print("Epoch %3d, Training Loss: %.4f" % (epoch + 1, epoch_loss))
                model.train()

        return self.model_parameters

    def train(self, model_parameters, id_list, curr_round, aggregate=False):
        """Train local model with different dataset according to :attr:`idx` in :attr:`id_list`.
        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            id_list (list[int]): Client id in this training serial.
            aggregate (bool): Whether to perform partial aggregation on this group of clients' local model in the end of local training round.
        Note:
            Normally, aggregation is performed by server, while we provide :attr:`aggregate` option here to perform
            partial aggregation on current client group. This partial aggregation can reduce the aggregation workload
            of server.
        Returns:
            Serialized model parameters / list of model parameters.
        """
        self.curr_round = curr_round  # set current training round
        param_list = []
        args = self.args
        self._LOGGER.info(
            "[Round {}] Local training with client id list: {}".format(curr_round, id_list))
        for idx in id_list:
            self._LOGGER.info(
                "[Round {}] Starting training procedure of client [{}]".format(curr_round, idx))

            data_loader = self._get_dataloader(client_id=idx)
            self._train_alone(model_parameters=model_parameters,
                              train_loader=data_loader,
                              client_id=idx,
                              client_sample_num=self.data_slices[idx].shape[0],
                              lr=args['lr'] * (args['lr_decay_per_round'] ** curr_round),
                              alpha_coef=self.sub_alpha_coef_adpt[idx],
                              weight_decay=args["weight_decay"])
            param_list.append(self.model_parameters)

        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list)
            return aggregated_parameters
        else:
            return param_list
