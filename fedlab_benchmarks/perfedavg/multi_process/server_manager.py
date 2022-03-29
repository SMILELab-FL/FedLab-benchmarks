import threading
import torch

from fedlab.core.communicator.processor import PackageProcessor, Package
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.utils.logger import Logger
from fedlab.utils.message_code import MessageCode


class PerFedAvgSyncServerManager(ServerSynchronousManager):
    def __init__(
        self,
        network,
        fedavg_handler,
        finetune_handler,
        personalization_handler,
        logger=Logger(),
    ):
        super(PerFedAvgSyncServerManager, self).__init__(network, None)
        self._LOGGER = logger
        self.finetune_handler = finetune_handler
        self.fedavg_handler = fedavg_handler
        self.personalization_handler = personalization_handler

    def shutdown(self):
        super(PerFedAvgSyncServerManager, self).shutdown()

    def main_loop(self):
        # FedAvg phase
        self._LOGGER.info("\033[1;33m===== FEDAVG PHASE =====\033[0m")
        while self.fedavg_handler.stop_condition() is False:
            allocated_clients = self.fedavg_handler.allocate_clients(
                self._network.world_size - 1
            )
            for rank, clients in enumerate(allocated_clients):
                activate = threading.Thread(
                    target=self.activate_process,
                    kwargs=dict(
                        rank=rank + 1,  # client's rank starts from 1
                        clients=torch.tensor(clients, dtype=torch.float),
                        flag=torch.tensor([1]),
                    ),
                )
                activate.start()
            done = False
            while not done:
                # payload: [weight_1, grads_1, weight_2, grads_2, ...] (grads are serialized)
                _, message_code, payload = PackageProcessor.recv_package()
                if message_code == MessageCode.ParameterUpdate:
                    for i in range(0, len(payload), 2):
                        if self.fedavg_handler.add_weight_and_grads(
                            payload[i], payload[i + 1]
                        ):
                            done = True
                else:
                    raise Exception("Unexpected message code {}".format(message_code))
        # Fine-tune phase
        if self.finetune_handler is not None:
            self._LOGGER.info("\033[1;33m===== FINE-TUNE PHASE =====\033[0m")
            while self.finetune_handler.stop_condition() is False:
                allocated_clients = self.finetune_handler.allocate_clients(
                    self._network.world_size - 1
                )
                for rank, clients in enumerate(allocated_clients):
                    activate = threading.Thread(
                        target=self.activate_process,
                        kwargs=dict(
                            rank=rank + 1,
                            clients=torch.tensor(clients, dtype=torch.float),
                            flag=torch.tensor([0], dtype=torch.float),
                        ),
                    )
                    activate.start()
                done = False
                while not done:
                    # payload: [grads_1, grads_2, ...] (grads are serialized)
                    _, message_code, payload = PackageProcessor.recv_package()
                    if message_code == MessageCode.ParameterUpdate:
                        for grads in payload:
                            if self.finetune_handler.add_grads(grads):
                                done = True
                    else:
                        raise Exception(
                            "Unexpected message code {}".format(message_code)
                        )

        # Personalization and final evaluation phase
        self._LOGGER.info("\033[1;33m===== PERSONALIZATION PHASE =====\033[0m")
        while self.personalization_handler.stop_condition() is False:
            allocated_clients = self.personalization_handler.allocate_clients(
                self._network.world_size - 1
            )
            for rank, clients in enumerate(allocated_clients):
                activate = threading.Thread(
                    target=self.activate_process,
                    kwargs=dict(
                        rank=rank + 1,
                        clients=torch.tensor(clients, dtype=torch.float),
                        flag=torch.tensor([-1], dtype=torch.float),
                    ),
                )
                activate.start()
            done = False
            while not done:
                # payload: [[init_loss_1, init_acc_1, per_loss_1, per_acc_1], ...]
                _, message_code, payload = PackageProcessor.recv_package()

                if message_code == MessageCode.ParameterUpdate:
                    for stats in payload:
                        if self.personalization_handler.add_stats(stats):
                            done = True
                else:
                    raise Exception("Unexpected message code {}".format(message_code))

    def activate_process(self, rank, clients, flag):
        if flag.item() == 1.0:
            model_parameters = self.fedavg_handler.model_parameters
        elif flag.item() == 0.0:
            model_parameters = self.finetune_handler.model_parameters
        else:
            model_parameters = self.personalization_handler.model_parameters
        pack = Package(
            message_code=MessageCode.ParameterUpdate,
            content=[flag.float(), clients, model_parameters],
        )
        PackageProcessor.send_package(pack, dst=rank)
