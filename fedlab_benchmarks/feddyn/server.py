import torch

import os
import sys

sys.path.append("../../../FedLab/")

from fedlab.core.network import DistNetwork
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import AverageMeter, load_dict
from fedlab.utils import SerializationTool, Aggregators

from config import local_grad_vector_file_pattern, clnt_params_file_pattern
import models


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
        "./Output/{}_{}_{}_{}.txt".format(config['partition'], config['network'],
                                          config['dataset'], config['run']), "w")

    key_name = ['avg_mdl_train', 'avg_mdl_test',
                'all_mdl_train', 'all_mdl_test',
                'cld_mdl_train', 'cld_mdl_test']

    record.write(str(config) + "\n")
    for key in key_name:
        record.write(f"{key}_acc:" + str(acces[key]) + "\n")
        record.write(f"{key}_loss:" + str(losses[key]) + "\n")
    record.close()


class FedDynServerHandler(SyncParameterServerHandler):
    def __init__(self,
                 test_loader,
                 train_loader,
                 global_round=5,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=None,
                 args=None):
        # get basic model
        model = getattr(models, args['model_name'])(args['model_name'])
        super().__init__(model,
                         global_round=global_round,
                         cuda=cuda,
                         sample_ratio=sample_ratio,
                         logger=logger)

        self.test_loader = test_loader
        self.train_loader = train_loader
        self.cld_mdl_test_loss, self.cld_mdl_test_acc = [], []
        self.cld_mdl_train_loss, self.cld_mdl_train_acc = [], []
        self.avg_mdl_test_loss, self.avg_mdl_test_acc = [], []
        self.avg_mdl_train_loss, self.avg_mdl_train_acc = [], []
        self.all_mdl_test_loss, self.all_mdl_test_acc = [], []
        self.all_mdl_train_loss, self.all_mdl_train_acc = [], []
        self.args = args
        self.local_param_list = []
        num_clients = args['num_clients']
        # file params initialization
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

        # =========== Evaluate model on train/test set
        avg_model = getattr(models, self.args['model_name'])(self.args['model_name'])
        SerializationTool.deserialize_model(avg_model, avg_mdl_param)

        all_model = getattr(models, self.args['model_name'])(self.args['model_name'])
        clnt_params_list = []
        for cid in range(self.client_num_in_total):
            clnt_params_file = clnt_params_file_pattern.format(cid=cid)
            curr_clnt_params = torch.load(clnt_params_file)
            clnt_params_list.append(curr_clnt_params)
        all_model_params = Aggregators.fedavg_aggregate(clnt_params_list)
        SerializationTool.deserialize_model(all_model, all_model_params)

        # evaluate on test set
        cld_mdl_test_loss, cld_mdl_test_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                                                       self.test_loader)
        avg_mdl_test_loss, avg_mdl_test_acc = evaluate(avg_model, torch.nn.CrossEntropyLoss(),
                                                       self.test_loader)
        all_mdl_test_loss, all_mdl_test_acc = evaluate(all_model, torch.nn.CrossEntropyLoss(),
                                                       self.test_loader)
        self.cld_mdl_test_loss.append(cld_mdl_test_loss)
        self.cld_mdl_test_acc.append(cld_mdl_test_acc)
        self.avg_mdl_test_loss.append(avg_mdl_test_loss)
        self.avg_mdl_test_acc.append(avg_mdl_test_acc)
        self.all_mdl_test_loss.append(all_mdl_test_loss)
        self.all_mdl_test_acc.append(all_mdl_test_acc)

        # evaluate on train set
        cld_mdl_train_loss, cld_mdl_train_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                                                         self.train_loader)
        avg_mdl_train_loss, avg_mdl_train_acc = evaluate(avg_model, torch.nn.CrossEntropyLoss(),
                                                         self.train_loader)
        all_mdl_train_loss, all_mdl_train_acc = evaluate(all_model, torch.nn.CrossEntropyLoss(),
                                                         self.train_loader)
        self.cld_mdl_train_loss.append(cld_mdl_train_loss)
        self.cld_mdl_train_acc.append(cld_mdl_train_acc)
        self.avg_mdl_train_loss.append(avg_mdl_train_loss)
        self.avg_mdl_train_acc.append(avg_mdl_train_acc)
        self.all_mdl_train_loss.append(all_mdl_train_loss)
        self.all_mdl_train_acc.append(all_mdl_train_acc)

        # write into file
        acces = {
            'avg_mdl_train': self.avg_mdl_train_acc,
            'avg_mdl_test': self.avg_mdl_test_acc,
            'all_mdl_train': self.all_mdl_train_acc,
            'all_mdl_test': self.all_mdl_test_acc,
            'cld_mdl_train': self.cld_mdl_train_acc,
            'cld_mdl_test': self.cld_mdl_test_acc
        }
        losses = {
            'avg_mdl_train': self.avg_mdl_train_loss,
            'avg_mdl_test': self.avg_mdl_test_loss,
            'all_mdl_train': self.all_mdl_train_loss,
            'all_mdl_test': self.all_mdl_test_loss,
            'cld_mdl_train': self.cld_mdl_train_loss,
            'cld_mdl_test': self.cld_mdl_test_loss
        }
        write_file(acces, losses, self.args)

        # =========== save model to file
        torch.save(self._model.state_dict(), os.path.join(self.args['out_dir'], "cld_model.pkl"))
        torch.save(avg_model.state_dict(), os.path.join(self.args['out_dir'], "avg_model.pkl"))
        torch.save(all_model.state_dict(), os.path.join(self.args['out_dir'], "all_model.pkl"))

        # =========== reset cache cnt
        self.cache_cnt = 0
        self.client_buffer_cache = []
        self.train_flag = False
