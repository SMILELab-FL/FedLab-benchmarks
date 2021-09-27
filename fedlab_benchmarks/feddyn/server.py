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


# to replace "from fedlab.core.server.scale.manager import ScaleSynchronousManager"
class ScaleSynchronousManager(ServerSynchronousManager):
    """ServerManager used in scale scenario."""

    def __init__(self, network, handler):
        super().__init__(network, handler)
        self.curr_round = -1

    def activate_clients(self):
        """Add client id map"""
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client Activation Procedure")
        for rank, values in rank_dict.items():
            self._LOGGER.info("rank {}, client ids {}".format(rank, values))

            # Send parameters
            param_pack = Package(message_code=MessageCode.ParameterUpdate,
                                 content=self._handler.model_parameters)
            PackageProcessor.send_package(package=param_pack, dst=rank)

            # Send activate id list
            id_list = torch.Tensor(values).int()
            act_pack = Package(message_code=MessageCode.ParameterUpdate,
                               content=id_list,
                               data_type=1)
            PackageProcessor.send_package(package=act_pack, dst=rank)

    def on_receive(self, sender, message_code, payload):
        if message_code == MessageCode.ParameterUpdate:
            for model_parameters in payload:
                update_flag = self._handler.add_model(sender, model_parameters)
                # update current round after handler adds model
                self.curr_round = self._hander.round
                if update_flag is True:
                    return update_flag
        else:
            raise Exception("Unexpected message code {}".format(message_code))


class FedDynServerHandler(SyncParameterServerHandler):
    pass
