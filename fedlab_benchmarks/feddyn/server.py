import argparse
import os

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
from fedlab.core.network import DistNetwork
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.network import DistNetwork
from fedlab.core.communicator import Package, PackageProcessor

from fedlab.utils.functional import AverageMeter
from fedlab.utils.functional import load_dict
from fedlab.utils.dataset import functional as dataF
from fedlab.utils import MessageCode, SerializationTool, Aggregators

from config import local_grad_vector_file_pattern, clnt_params_file_pattern


def evaluate(model, criterion, test_loader):
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.sum, acc_.avg


def write_file(acces, losses, config):
    record = open(
        "{}_{}_{}_{}.txt".format(config['partition'], config['network'],
                                 config['dataset'], config['run']), "w")

    record.write(str(config) + "\n")
    record.write(str(losses) + "\n")
    record.write(str(acces) + "\n")
    record.close()


class RecodeHandler(SyncParameterServerHandler):
    def __init__(self,
                 model,
                 test_loader,
                 global_round=5,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=None,
                 config=None):
        super().__init__(model,
                         global_round=global_round,
                         cuda=cuda,
                         sample_ratio=sample_ratio,
                         logger=logger)

        self.test_loader = test_loader
        self.loss_ = []
        self.acc_ = []
        self.config = config

    def _update_model(self, model_parameters_list):
        super()._update_model(model_parameters_list)

        loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                             self.test_loader)

        self.loss_.append(loss)
        self.acc_.append(acc)

        write_file(self.acc_, self.loss_, self.config)


class FedDynServerHandler(SyncParameterServerHandler):
    def __init__(self,
                 model,
                 test_loader,
                 global_round=5,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=None,
                 args=None):
        super().__init__(model,
                         global_round=global_round,
                         cuda=cuda,
                         sample_ratio=sample_ratio,
                         logger=logger)

        self.test_loader = test_loader
        self.loss_ = []
        self.acc_ = []
        self.args = args
        self.local_param_list = []
        num_clients = args['num_clients']
        serialized_params = SerializationTool.serialize_model(model)
        zeros_params = torch.zeros(serialized_params.shape[0])
        for cid in range(num_clients):
            local_grad_vector_file = local_grad_vector_file_pattern.format(cid=cid)
            clnt_params_file = clnt_params_file_pattern.format(cid=cid)
            torch.save(zeros_params, local_grad_vector_file)
            torch.save(serialized_params, clnt_params_file)

    def _update_model(self, model_parameters_list):
        self._LOGGER.info(
            "Model parameters aggregation, number of aggregation elements {}".format(
                len(model_parameters_list)))
        # =========== update server model
        avg_mdl_param = Aggregators.fedavg_aggregate(model_parameters_list)
        # avg_local_grad is avg of local gradients from all clients
        # read serialized params of all clients from local files and average them
        local_grad_vector_list = []
        for cid in range(self.client_num_in_total):
            local_grad_vector_file = local_grad_vector_file_pattern.format(cid=cid)
            curr_local_grad_vector = torch.load(local_grad_vector_file)
            local_grad_vector_list.append(curr_local_grad_vector)
        avg_local_grad = Aggregators.fedavg_aggregate(local_grad_vector_list)
        cld_mdl_param = avg_mdl_param + avg_local_grad
        # load latest cloud model params into server model
        SerializationTool.deserialize_model(self._model, cld_mdl_param)

        # =========== reset cache cnt
        self.cache_cnt = 0
        self.client_buffer_cache = []
        self.train_flag = False
