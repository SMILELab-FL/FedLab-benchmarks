import argparse

from fedlab.utils.functional import evaluate
from fedlab.utils import Logger, SerializationTool, Aggregators
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.network import DistNetwork
from setting import get_model, get_dataset
import torch
import torchvision
from torchvision import transforms
from copy import deepcopy

class TestHandler(SyncParameterServerHandler):

    def __init__(self,
                 model,
                 global_round,
                 sample_ratio,
                 cuda=False,
                 logger=None):
        super().__init__(model, global_round, sample_ratio, cuda, logger)

        testset = torchvision.datasets.MNIST(root='../datasets/mnist/',
                                             train=False,
                                             download=True,
                                             transform=transforms.ToTensor())

        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=int(
                                                          len(testset) / 10),
                                                      drop_last=False,
                                                      shuffle=False)

    def _update_global_model(self, payload):
        
        assert torch.equal(payload[0], self.model_parameters)
        self.client_buffer_cache.append(deepcopy(payload[0]))
 

        if len(self.client_buffer_cache) == self.client_num_per_round:

            assert torch.equal(model_parameters_list[0], model_parameters_list[1])
            

            model_parameters_list = self.client_buffer_cache
            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(
                model_parameters_list)

            assert torch.equal(serialized_parameters, self.model_parameters)
        
            SerializationTool.deserialize_model(self._model,
                                                serialized_parameters)

            loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                                 self.testloader)
            self._LOGGER.info(
                "check check Server evaluate loss {:.4f}, acc {:.4f}".format(
                    loss, acc))

            self.round += 1

            # reset cache cnt
            self.cache_cnt = 0
            self.client_buffer_cache = []

            return True
        else:
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    model = get_model(args)
    LOGGER = Logger(log_name="server", log_file="test.txt")
    handler = TestHandler(model,
                          global_round=args.round,
                          logger=LOGGER,
                          sample_ratio=args.sample)
    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0,
                          ethernet=args.ethernet)
    manager_ = SynchronousServerManager(handler=handler,
                                        network=network,
                                        logger=LOGGER)
    manager_.run()
