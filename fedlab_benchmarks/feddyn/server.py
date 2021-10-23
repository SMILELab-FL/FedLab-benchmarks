import torch

import os
import numpy as np
import sys

sys.path.append("../../../FedLab/")

from fedlab.core.network import DistNetwork
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.core.communicator.package import Package
from fedlab.core.communicator.processor import PackageProcessor
from fedlab.utils.functional import AverageMeter, load_dict
from fedlab.utils.message_code import MessageCode
from fedlab.utils import SerializationTool, Aggregators, Logger

from config import local_grad_vector_file_pattern, clnt_params_file_pattern
import models


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k, in range of [0, 1]"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

# # using accuracy function
# def evaluate2(model, criterion, test_loader):
#     model.eval()
#     gpu = next(model.parameters()).device

#     loss_ = AverageMeter()
#     acc_ = AverageMeter()

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             batch_size = len(labels)
#             inputs = inputs.to(gpu)
#             labels = labels.to(gpu)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             acc1 = accuracy(outputs, labels, topk=(1,))

#             loss_.update(loss.item(), batch_size)
#             acc_.update(acc1[0].item(), batch_size)

#     return loss_.avg, acc_.avg


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

    return loss_.avg, acc_.avg


def write_file(acces, losses, config):
    record = open(
        "./Output/{}_{}_{}.txt".format(config['partition'], config['model_name'],
                                       config['dataset']), "w")

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
            local_grad_vector_file = os.path.join(self.args['out_dir'],
                                                  local_grad_vector_file_pattern.format(cid=cid))
            clnt_params_file = os.path.join(self.args['out_dir'],
                                            clnt_params_file_pattern.format(cid=cid))
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
            local_grad_vector_file = os.path.join(self.args['out_dir'],
                                                  local_grad_vector_file_pattern.format(cid=cid))
            curr_local_grad_vector = torch.load(local_grad_vector_file)
            local_grad_vector_list.append(curr_local_grad_vector)
        avg_local_grad = Aggregators.fedavg_aggregate(local_grad_vector_list)

        cld_mdl_param = avg_mdl_param + avg_local_grad
        # load latest cloud model params into server model
        SerializationTool.deserialize_model(self._model, cld_mdl_param)
        self._LOGGER.info("Server model update DONE")

        # =========== Evaluate model on train/test set
        avg_model = getattr(models, self.args['model_name'])(self.args['model_name'])
        SerializationTool.deserialize_model(avg_model, avg_mdl_param)

        all_model = getattr(models, self.args['model_name'])(self.args['model_name'])
        clnt_params_list = []
        for cid in range(self.client_num_in_total):
            clnt_params_file = os.path.join(self.args['out_dir'],
                                            clnt_params_file_pattern.format(cid=cid))
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
        self._LOGGER.info("Server model evaluation on test set done")

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
        self._LOGGER.info("Server model evaluation on train set done")

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
        self.write_file(acces, losses, self.args)

        # =========== save model to file
        torch.save(self._model.state_dict(), os.path.join(self.args['out_dir'], "cld_model.pkl"))
        torch.save(avg_model.state_dict(), os.path.join(self.args['out_dir'], "avg_model.pkl"))
        torch.save(all_model.state_dict(), os.path.join(self.args['out_dir'], "all_model.pkl"))
        self._LOGGER.info("Server model save done")

        # =========== reset cache cnt
        self.cache_cnt = 0
        self.client_buffer_cache = []
        self.train_flag = False

    def write_file(self, acces, losses, config):
        record = open(
            "./Output/{}_{}_{}.txt".format(config['partition'], config['model_name'],
                                           config['dataset']), "w")

        key_name = ['avg_mdl_train', 'avg_mdl_test',
                    'all_mdl_train', 'all_mdl_test',
                    'cld_mdl_train', 'cld_mdl_test']

        record.write(str(config) + "\n")
        for key in key_name:
            record.write(f"{key}_acc:" + str(acces[key]) + "\n")
            record.write(f"{key}_loss:" + str(losses[key]) + "\n")
        record.close()


class FedAvgServerHandler(SyncParameterServerHandler):
    def __init__(self, test_loader,
                 weight_list=None,
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
        self.args = args
        self.weight_list = weight_list
        self.client_this_round = []
        self.acc_ = []
        self.loss_ = []

    def _update_model(self, model_parameters_list):
        self._LOGGER.info(
            "Model parameters aggregation, number of aggregation elements {}".
                format(len(model_parameters_list)))
        # use aggregator
        curr_weight_sum = sum([self.weight_list[cid] for cid in self.client_this_round])
        # self._LOGGER.info(f"curr_weight_sum: {curr_weight_sum}")
        serialized_parameters = Aggregators.fedavg_aggregate(
            model_parameters_list) * len(self.client_this_round) / curr_weight_sum
        SerializationTool.deserialize_model(self._model, serialized_parameters)

        # evaluate on test set
        test_loss, test_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                                       self.test_loader)
        self.acc_.append(test_acc)
        self.loss_.append(test_loss)
        self.write_file()

        # reset cache cnt
        self.cache_cnt = 0
        self.client_buffer_cache = []
        self.train_flag = False

    def add_model(self, sender_rank, model_parameters):
        self.client_buffer_cache.append(model_parameters.clone())
        self.cache_cnt += 1

        # cache is full
        if self.cache_cnt == self.client_num_per_round:
            self._update_model(self.client_buffer_cache)
            self.round += 1
            return True
        else:
            return False

    def write_file(self):
        file_name = os.path.join(self.args['out_dir'],
                                 f"FedAvg_{self.args['model_name']}_{self.args['partition']}_{self.args['dataset']}.txt")
        record = open(file_name, "w")

        record.write(str(self.args) + "\n")
        record.write(f"acc:" + str(self.acc_) + "\n")
        record.write(f"loss:" + str(self.loss_) + "\n")
        record.close()


class FedAvgServerManager(ScaleSynchronousManager):
    def __init__(self, network, handler, logger=Logger()):
        super(FedAvgServerManager, self).__init__(network, handler, logger)
        self.client_this_round = []

    def activate_clients(self):
        """Use client id mapping: Coordinator.
        Here we use coordinator to find the rank client process with specific client_id.
        """
        clients_this_round = self._handler.sample_clients()
        self._handler.client_this_round = clients_this_round  # server handler

        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client Activation Procedure")
        for rank, values in rank_dict.items():
            self._LOGGER.info("rank {}, client ids {}".format(rank, values))

            # Send parameters
            param_pack = Package(message_code=MessageCode.ParameterUpdate,
                                 content=self._handler.model_parameters)
            PackageProcessor.send_package(package=param_pack, dst=rank)

            # Send activate id list
            id_list = torch.Tensor(values).to(torch.int32)
            act_pack = Package(message_code=MessageCode.ParameterUpdate,
                               content=id_list)
            PackageProcessor.send_package(package=act_pack, dst=rank)
