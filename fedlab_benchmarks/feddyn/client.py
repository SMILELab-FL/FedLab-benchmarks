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
from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork

from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict
from fedlab.utils.dataset import functional as dataF



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

    def train(self, model_parameters, id_list, aggregate=False):
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
        param_list = []
        args = self.args
        self._LOGGER.info(
            "Local training with client id list: {}".format(id_list))
        for idx in id_list:
            self._LOGGER.info(
                "Starting training procedure of client [{}]".format(idx))

            data_loader = self._get_dataloader(client_id=idx)
            self._train_alone(model_parameters=model_parameters,
                              train_loader=data_loader,
                              client_id=idx,
                              client_sample_num=self.data_slices[idx].shape[0],
                              lr=args['lr'] * (args['lr_decay_per_round'] ** comm_round),
                              alpha_coef=self.sub_alpha_coef_adpt[idx],
                              weight_decay=args["weight_decay"])
            param_list.append(self.model_parameters)

        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list)
            return aggregated_parameters
        else:
            return param_list


