import argparse
import threading
from fedlab.utils import Logger, SerializationTool, MessageCode
from fedlab.utils.functional import evaluate
from fedlab.core.communicator import PackageProcessor
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork
from setting import get_model, get_dataset
import numpy as np
import torch
import torchvision
from torchvision import transforms


class qfedavgServerHandler(SyncParameterServerHandler):

    def __init__(self,
                 model,
                 global_round=5,
                 cuda=False,
                 sample_ratio=1,
                 logger=Logger()):
        super().__init__(model, global_round, cuda, sample_ratio, logger)
        self.local_losses = []
        self.lr = 0.01
        self.q = 10

        testset = torchvision.datasets.MNIST(root='../../datasets/mnist/',
                                             train=False,
                                             download=True,
                                             transform=transforms.ToTensor())

        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=int(
                                                          len(testset) / 10),
                                                      drop_last=False,
                                                      shuffle=False)

    def _update_model(self, model_parameters_list, client_loss):
        gradients = [(self.model_parameters - parameters) / self.lr
                     for parameters in model_parameters_list]
        Deltas = [
            grad * np.float_power(loss + 1e-10, (self.q - 1))
            for grad, loss in zip(gradients, client_loss)
        ]
        hs = [
            self.q * np.float_power(loss + 1e-10,
                                    (self.q - 1)) * (grad.norm()**2) +
            1.0 / self.lr * np.float_power(loss + 1e-10, self.q)
            for grad, loss in zip(gradients, client_loss)
        ]


        print(Deltas[0].shape)
        demominator = np.sum(np.asarray(hs))
        scaled_deltas = [delta / demominator for delta in Deltas]
        updates = torch.Tensor(sum(scaled_deltas))
        print(updates.shape)
        model_parameters = self.model_parameters - updates

        SerializationTool.deserialize_model(self._model, model_parameters)

        loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                             self.testloader)
        self._LOGGER.info("check check Server evaluate loss {:.4f}, acc {:.4f}".format(
            loss, acc))

        self.cache_cnt = 0

    def add_model(self, sender_rank, model_parameters, local_loss):
        self.local_losses.append(local_loss.item())
        self.client_buffer_cache.append(model_parameters.clone())
        self.cache_cnt += 1

        # cache is full
        if self.cache_cnt == self.client_num_per_round:
            self._update_model(self.client_buffer_cache, self.local_losses)
            self.round += 1
            return True
        else:
            return False

class qfedavgServerManager(ServerSynchronousManager):

    def main_loop(self):
        while self._handler.stop_condition() is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()
            while True:
                sender, message_code, payload = PackageProcessor.recv_package()
                if message_code == MessageCode.ParameterUpdate:
                    if self._handler.add_model(sender, payload[0], payload[1]):
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    model = get_model(args)
    LOGGER = Logger(log_name="server")
    handler = qfedavgServerHandler(model,
                                         global_round=args.round,
                                         logger=LOGGER,
                                         sample_ratio=args.sample)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0,
                          ethernet=args.ethernet)

    manager_ = qfedavgServerManager(handler=handler,
                                        network=network,
                                        logger=LOGGER)
    manager_.run()
