import torch
from fedlab.utils.message_code import MessageCode
from fedlab.core.communicator import Package, PackageProcessor
from fedlab.core.client.manager import ClientPassiveManager


class PerFedAvgClientManager(ClientPassiveManager):
    def __init__(self, network, fedavg_trainer, fine_tuner):
        super(PerFedAvgClientManager, self).__init__(network, fedavg_trainer)
        self.output = []
        self._fine_tuner = fine_tuner

        """
        "1" for performing FedAvg;
        "0" for performing Reptile;
        "-1" for performing Personalizaiton and evaluation.
        """
        self.flag = 1

    def main_loop(self):
        while True:
            _, message_code, payload = PackageProcessor.recv_package(
                src=0
            )  # payload: [flag, client_id, model_parameters], all entries are torch.Tensor
            if message_code == MessageCode.Exit:
                break
            elif message_code == MessageCode.ParameterUpdate:
                flag, clients, model_parameters = payload
                client_id_list = clients.int().numpy().tolist()
                self.flag = flag.int().item()
                print(
                    "process [{}] have received tasks: {}".format(
                        self._network.rank, client_id_list
                    )
                )
                if self.flag == 1:  # FedAvg
                    for client_id in client_id_list:
                        self.output.append(
                            self._trainer.train(client_id, model_parameters)
                        )
                elif self.flag == 0:  # Fine-tune(Reptile)
                    for client_id in client_id_list:
                        self.output.append(
                            self._fine_tuner.train(
                                client_id, model_parameters, validation=False
                            )
                        )
                elif self.flag == -1:  # Personalizaiton and final evaluation
                    for client_id in client_id_list:
                        self.output.append(
                            self._trainer.evaluate(client_id, model_parameters)
                        )
                self.synchronize()
            else:
                raise ValueError(
                    "Invalid MessageCode {}. Please see MessageCode Enum".format(
                        message_code
                    )
                )

    def synchronize(self):
        compressed_output = []
        if self.flag == 1:
            # FedAvg's output includes weight and grads
            # weight: torch.Tensor; grads: List[torch.Tensor]
            for weight, grads in self.output:
                serialized_grads = torch.cat(
                    [grad.detach().reshape(-1) for grad in grads]
                ).cpu()
                compressed_output.append(weight.float().unsqueeze(0))
                compressed_output.append(serialized_grads)
            # compressed_output: [weight_1, grad_1, weight_2, grad_2, ...]
        elif self.flag == 0:
            # Reptile's output only includes grads
            for grads in self.output:
                serialized_grads = torch.cat(
                    [grad.detach().reshape(-1) for grad in grads]
                ).cpu()
                compressed_output.append(serialized_grads)
            # compressed_output: [grad_1, grad_2, ...]
        else:
            # evaluate returns (init_loss, init_acc), (per_loss, per_acc)
            for init_stat, per_stat in self.output:
                init_stat = torch.tensor(init_stat, dtype=torch.float)
                per_stat = torch.tensor(per_stat, dtype=torch.float)
                compressed_output.append(torch.cat([init_stat, per_stat]))
            # compressed_output: [[init_loss_1, init_acc_1, per_loss_1, per_acc_1], [init_loss_2, ...]
        pack = Package(
            message_code=MessageCode.ParameterUpdate, content=compressed_output
        )
        PackageProcessor.send_package(pack, dst=0)
        self.output = []
