from sys import path

path.append("../")

import random
import torch
from fedlab.core.server.handler import ParameterServerBackendHandler
from fedlab.utils.logger import Logger
from utils import get_optimizer


class PersonalizaitonHandler(ParameterServerBackendHandler):
    """handle personalization and final evaluation

    Args:
        model (torch.nn.Module): Global model.
        global_round (int): Usually set as 1, it's okay to set > 1 for testing more round.
        client_num_in_total (int): Quantity of clients.
        client_num_per_round (int): Client num of participating per and eval in each global round
        cuda (bool, optional): True For using GPUs. Defaults to False.
        logger (fedelab.utils.Logger, optional): Logger for logging. Defaults to Logger().
    """

    def __init__(
        self,
        model,
        global_round,
        client_num_in_total,
        client_num_per_round,
        cuda=False,
        logger=Logger(log_name="pers"),
    ):
        super(PersonalizaitonHandler, self).__init__(model, cuda)
        self.client_num_in_total = client_num_in_total
        self.client_num_per_round = client_num_per_round
        self.global_round = global_round
        self.round = 0
        self.client_stats_cache = []
        self.avg_stats_cache = []
        self.logger = logger

    def stop_condition(self) -> bool:
        if self.round >= self.global_round:
            final_init_loss = final_init_acc = final_per_loss = final_per_acc = 0
            for stat in self.avg_stats_cache:
                # stat = [init_avg_loss, init_avg_acc, per_avg_loss, per_avg_acc]
                final_init_loss += stat[0]
                final_init_acc += stat[1]
                final_per_loss += stat[2]
                final_per_acc += stat[3]
            final_init_loss /= self.round
            final_init_acc /= self.round
            final_per_loss /= self.round
            final_per_acc /= self.round
            self.logger.info(
                "\033[1;33m\ninit_loss: {:.4f}\ninit_acc: {:.2f}%\nper_loss: {:.4f}\nper_acc: {:.2f}%\033[0m".format(
                    final_init_loss,
                    (final_init_acc * 100.0),
                    final_per_loss,
                    (final_per_acc * 100.0),
                )
            )
            return True
        return False

    def allocate_clients(self, n):
        """Evenly allocate tasks to each node(process).

        Args:
            n (int): num of nodes(process)

        Returns:
            List[List]: each list includes client's IDs for corresponding process.
        """
        selected_clients = random.sample(
            range(self.client_num_in_total), self.client_num_per_round,
        )
        self.logger.info(
            "selected clients in round [{}]: {}".format(self.round, selected_clients)
        )
        return _allocate_clients(selected_clients, n)

    def add_stats(self, stats):
        """append client's evaluating results(init/per) to buffer and display average results finally

        Args:
            stats (torch.Tensor): [init_loss, init_acc, per_loss, per_acc]

        Returns:
            bool: if already collected all clients's stats, return True
        """
        self.client_stats_cache.append(stats)

        if len(self.client_stats_cache) == self.client_num_per_round:
            init_avg_loss = init_avg_acc = per_avg_loss = per_avg_acc = 0
            for stats in self.client_stats_cache:
                init_loss, init_acc, per_loss, per_acc = torch.split(stats, 1)
                init_avg_loss += init_loss.item()
                init_avg_acc += init_acc.item()
                per_avg_loss += per_loss.item()
                per_avg_acc += per_acc.item()
            init_avg_loss /= self.client_num_per_round
            init_avg_acc /= self.client_num_per_round
            per_avg_loss /= self.client_num_per_round
            per_avg_acc /= self.client_num_per_round

            self.avg_stats_cache.append(
                [init_avg_loss, init_avg_acc, per_avg_loss, per_avg_acc]
            )
            self.client_stats_cache = []
            self.round += 1
            return True
        return False


class FineTuneHandler(ParameterServerBackendHandler):
    """Handle fine-tuning procedure

    Args:
        model (torch.nn.Module): Global Model
        global_round (int): Fine-tuning communication round
        client_num_in_total (int): Quantity of clients.
        client_num_per_round (int): Client num of participating per and eval in each global round
        optimizer_type (str): Declare type of server optimizer
        optimizer_args (dict): Provide necessary args for generating server optimizer
        cuda (bool, optional): True for using GPUs. Defaults to False.
        logger (fedlab.utils.Logger, optional): Defaults to Logger().
    """

    def __init__(
        self,
        model,
        global_round,
        client_num_in_total,
        client_num_per_round,
        optimizer_type,
        optimizer_args,
        cuda=False,
        logger=Logger(log_name="fine-tune"),
    ):
        super(FineTuneHandler, self).__init__(model, cuda)
        self.client_num_in_total = client_num_in_total
        self.client_num_per_round = client_num_per_round
        self.global_round = global_round
        self.round = 0
        self.optimizer = get_optimizer(self._model, optimizer_type, optimizer_args)
        self.client_gradients_buffer_cache = []
        self.logger = logger

    def stop_condition(self) -> bool:
        return self.round >= self.global_round

    def allocate_clients(self, n):
        selected_clients = random.sample(
            range(self.client_num_in_total), self.client_num_per_round
        )
        self.logger.info(
            "selected clients in round [{}]: {}".format(self.round, selected_clients)
        )
        return _allocate_clients(selected_clients, n)

    def add_grads(self, grads):
        """append client grads to buffer

        Args:
            grads (torch.Tensor): serialized gradients

        Returns:
            bool: return True if collected all clients's grads
        """
        self.client_gradients_buffer_cache.append(grads)
        if len(self.client_gradients_buffer_cache) == self.client_num_per_round:
            self._update_model()
            self.round += 1
            return True
        return False

    def _update_model(self):
        self.optimizer.zero_grad()
        for grads in self.client_gradients_buffer_cache:
            _deserialize_gradients(self._model, grads / self.client_num_per_round)
        for param in self._model.parameters():
            param.grad.data.div_(self.client_num_per_round)
        self.optimizer.step()
        self.client_gradients_buffer_cache = []


class FedAvgHandler(ParameterServerBackendHandler):
    """Handle FedAvg training procedure

    Args:
        model (torch.nn.Module): Global Model
        global_round (int): Fine-tuning communication round
        client_num_in_total (int): Quantity of clients.
        client_num_per_round (int): Client num of participating per and eval in each global round
        optimizer_type (str): Declare type of server optimizer
        optimizer_args (dict): Provide necessary args for generating server optimizer
        cuda (bool, optional): True for using GPUs. Defaults to False.
        logger (fedlab.utils.Logger, optional): Defaults to Logger().
    """

    def __init__(
        self,
        model,
        global_round,
        client_num_in_total,
        optimizer_type,
        optimizer_args,
        client_num_per_round,
        cuda=False,
        logger=Logger(log_name="fedavg"),
    ):
        super(FedAvgHandler, self).__init__(model, cuda)
        self.client_num_in_total = client_num_in_total
        self.client_num_per_round = client_num_per_round
        self.global_round = global_round
        self.round = 0
        self.optimizer = get_optimizer(self._model, optimizer_type, optimizer_args)
        self.client_weights_buffer_cache = []
        self.client_gradients_buffer_cache = []
        self.cache_count = 0
        self.logger = logger

    def stop_condition(self) -> bool:
        return self.round >= self.global_round

    def allocate_clients(self, n):
        selected_clients = random.sample(
            range(self.client_num_in_total), self.client_num_per_round
        )
        self.logger.info(
            "selected clients in round [{}]: {}".format(self.round, selected_clients)
        )
        return _allocate_clients(selected_clients, n)

    def add_weight_and_grads(self, weight, gradients):
        """Append each client's weight and gradients to buffer

        Args:
            weight (torch.Tensor): Has shape: torch.Size([])
            gradients (torch.Tensor): Serialized gradients

        Returns:
            bool: Return True if finished update
        """
        self.client_weights_buffer_cache.append(weight)
        self.client_gradients_buffer_cache.append(gradients)

        if (
            len(self.client_gradients_buffer_cache)
            == len(self.client_weights_buffer_cache)
            == self.client_num_per_round
        ):
            self._update_model()
            self.round += 1
            return True
        return False

    def _update_model(self):
        self.optimizer.zero_grad()
        weights_sum = sum(self.client_weights_buffer_cache)
        all_client_weights = [
            weight / weights_sum for weight in self.client_weights_buffer_cache
        ]
        for weight, grads in zip(
            all_client_weights, self.client_gradients_buffer_cache
        ):
            _deserialize_gradients(self._model, weight * grads)
        self.optimizer.step()
        self.client_gradients_buffer_cache = []
        self.client_weights_buffer_cache = []


def _deserialize_gradients(model: torch.nn.Module, serialized_gradients: torch.Tensor):
    """Assigns serialized gradients to model.
    
    Args:
        model (torch.nn.Module): model to deserialize.
        serialized_gradients (torch.Tensor): serialized model gradients.
    """
    current_index = 0  # keep track of where to read from grad_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.size()
        if parameter.grad is None:
            parameter.grad = torch.zeros(
                size, requires_grad=True, device=parameter.device
            )
        parameter.grad.data.add_(
            serialized_gradients[current_index : current_index + numel]
            .view(parameter.grad.size())
            .to(parameter.device)
        )

        current_index += numel


def _allocate_clients(selected_clients, n):
    """Allocate selected clients at each round to all processes as evenly as possible

    Args:
        selected_clients (List[int]): Selected clients ID  
        n (int): Num of processes

    Returns:
        List[List[int]]: Allocated clients ID for each process.
    """
    client_num = len(selected_clients)
    step = int(client_num / n)
    allocated_clients = [
        selected_clients[i : i + step] for i in range(0, client_num, step)
    ]
    if client_num % n != 0:
        allocated_clients[-2] = allocated_clients[-2] + allocated_clients[-1]
        allocated_clients.pop(-1)
    return allocated_clients

