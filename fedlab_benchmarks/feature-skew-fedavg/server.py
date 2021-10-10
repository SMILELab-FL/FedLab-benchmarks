import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

torch.manual_seed(0)

import sys
sys.path.append("../../../FedLab")

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.core.communicator.processor import Package, PackageProcessor
from fedlab.core.coordinator import Coordinator
from fedlab.utils.functional import AverageMeter
from fedlab.utils.message_code import MessageCode

from models import SimpleCNNMNIST
from config import fmnist_noise_baseline_config

# class ServerLogManager(ScaleSynchronousManager):
#     def setup(self):
#         self._LOGGER.info("Server setup=====")
#         self._network.init_network_connection()

#         rank_client_id_map = {}

#         for rank in range(1, self._network.world_size):
#             _, _, content = PackageProcessor.recv_package(src=rank)
#             rank_client_id_map[rank] = content[0].item()
#             self._LOGGER.info(f"from client rank {rank}: {rank_client_id_map[rank]}")
        
#         self.coordinator = Coordinator(rank_client_id_map)
#         if self._handler is not None:
#             self._handler.client_num_in_total = self.coordinator.total
        
#         self._LOGGER.info(f"Client rank info receive done")
#         self._LOGGER.info(f"rank_client_id_map: {rank_client_id_map}")
#         self._LOGGER.info(f"server coordinator: {self.coordinator}")

#     def activate_clients(self):
#         """Use client id mapping: Coordinator. 
#         Here we use coordinator to find the rank client process with specific client_id.
#         """
#         clients_this_round = self._handler.sample_clients()
#         rank_dict = self.coordinator.map_id_list(clients_this_round)

#         self._LOGGER.info("Client Activation Procedure")
#         for rank, values in rank_dict.items():
#             self._LOGGER.info("rank {}, client ids {}".format(rank, values))

#             # Send parameters
#             param_pack = Package(message_code=MessageCode.ParameterUpdate,
#                                  content=self._handler.model_parameters)
#             PackageProcessor.send_package(package=param_pack, dst=rank)

#             # Send activate id list
#             id_list = torch.Tensor(values).int()
#             act_pack = Package(message_code=MessageCode.ParameterUpdate,
#                                content=id_list,
#                                data_type=1)
#             self._LOGGER.info(f"Try to send Act_id for rank {rank}: {id_list}")
#             PackageProcessor.send_package(package=act_pack, dst=rank)
#             self._LOGGER.info(f"Act_id for rank {rank} sent done: {id_list}")



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
        "noise_{}_{}_{}_{}.txt".format(config['partition'], config['network'],
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


# python server.py --world_size 2
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedAvg server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3003")
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--ethernet', type=str, default=None)

    parser.add_argument('--setting', type=str, default='noise')
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()

    model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)

    config = fmnist_noise_baseline_config
    config['run'] = args.run

    transform_test = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.FashionMNIST(
        root='../../../datasets/FMNIST/',
        train=False,
        download=True,
        transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config['test_batch_size'],
                                             drop_last=False,
                                             shuffle=False)

    handler = RecodeHandler(model,
                            global_round=config["round"],
                            sample_ratio=config["sample_ratio"],
                            test_loader=testloader,
                            cuda=True,
                            config=config)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler)
    manager_.run()
