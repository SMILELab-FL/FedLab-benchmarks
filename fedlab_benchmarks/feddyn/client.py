import argparse
import os
import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

import sys

sys.path.append("../../../FedLab/")

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


def save_model_params(model, file, logger=None):
    """

    Args:
        model (nn.Module): model to serialize and save.
        file (str): full path file name, ``*.pt`` is preferred.
        logger (Logger): logger for information print.

    Returns:

    """
    serialized_params = SerializationTool.serialize_model(model)
    torch.save(serialized_params, file)
    logger.info(f"{file} saved.")


class FedDynSerialTrainer(SubsetSerialTrainer):
    def __init__(self, model,
                 dataset,
                 data_slices,
                 client_weights=None,
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
        self.client_weights = client_weights
        self.round = 0  # global round, try not to use package to inform global round

    def _train_alone(self, cld_model_params,
                     train_loader,
                     client_id,
                     alpha_coef,
                     avg_mdl_param,
                     local_grad_vector):
        """
        cld_model_params: serialized model params from cloud model (server model)
        train_loader:
        client_id:
        alpha_coef:
        avg_mdl_param:  model avg of selected clients from last round
        local_grad_vector:
        """
        orig_lr = self.args['lr']
        lr_decay_per_round = self.args['lr_decay_per_round']
        lr = orig_lr * (lr_decay_per_round ** self.round)  # using learning rate decay
        weight_decay = self.args['weight_decay']
        epochs = self.args['epochs']
        max_norm = self.args['max_norm']
        print_freq = self.args['print_freq']

        SerializationTool.deserialize_model(self._model, cld_model_params)  # load model params
        loss_fn = torch.nn.CrossEntropyLoss()
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
                # equal to CrossEntropyLoss(reduction='sum') / targets.shape[0]
                loss_f_i = loss_fn(y_pred, targets.long())

                # Get linear penalty on the current parameter estimates
                # Note: DO NOT use SerializationTool.serialize_model() to serialize model params
                # here, they get same numeric result but result from SerializationTool doesn't
                # have 'grad_fn=<CatBackward>' !!!!!
                local_par_list = None
                for param in self.model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = alpha_coef * torch.sum(
                    local_par_list * (-avg_mdl_param + local_grad_vector))
                loss = loss_f_i + loss_algo

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=max_norm)  # Clip gradients
                optimizer.step()
                epoch_loss += loss.item() * targets.shape[0]

            if (e + 1) % print_freq == 0:
                epoch_loss /= len(self.data_slices[client_id])
                if weight_decay is not None:
                    # Add L2 loss to complete f_i
                    serialized_params = SerializationTool.serialize_model(self.model).numpy()
                    epoch_loss += (alpha_coef + weight_decay) / 2 * np.dot(serialized_params,
                                                                           serialized_params)
                self._LOGGER.info(
                    f"Client {client_id}, Epoch {e + 1}/{epochs}, Training Loss: {epoch_loss:.4f}")

        self._LOGGER.info(f"_train_alone(): Client {client_id}, Global Round {self.round} DONE")

        return self.model_parameters

    def train(self, model_parameters, id_list, aggregate=False):
        param_list = []

        self._LOGGER.info(
            "Local training with client id list: {}".format(id_list))
        for idx in id_list:
            self._LOGGER.info(
                "train(): Starting training procedure of client [{}]".format(idx))

            data_loader = self._get_dataloader(client_id=idx)
            alpha_coef_adpt = self.args['alpha_coef'] / self.client_weights[idx]
            self._train_alone(cld_model_params=model_parameters,
                              train_loader=data_loader,
                              client_id=idx,
                              alpha_coef=alpha_coef_adpt,
                              avg_mdl_param=avg_mdl_param,
                              local_grad_vector=local_grad_vector)
            param_list.append(self.model_parameters)

        self._LOGGER.info(f"train(): Serial Trainer Global Round {self.round} done")
        self.round += 1  # trainer global round counter update

        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list)
            return aggregated_parameters
        else:
            return param_list
