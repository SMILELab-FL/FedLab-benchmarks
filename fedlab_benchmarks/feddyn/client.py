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
    def __init__(self, model, client_num,
                 aggregator=None,
                 cuda=True,
                 logger=Logger('FedDyn-Client-Trainer'),
                 config=None):
        super().__init__(model,
                         client_num,
                         aggregator=None,
                         cuda=True,
                         logger=logger)
        lr = config['lr']

    pass
