import torch
from copy import deepcopy
from fedlab.utils.serialization import SerializationTool
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.utils.functional import evaluate
from fedlab.utils.logger import Logger
from torch.utils.data import DataLoader, random_split
from utils import get_optimizer
from tqdm import trange

from sys import path

path.append("../")

from leaf.pickle_dataset import PickleDataset


class PerFedAvgTrainer(ClientSGDTrainer):
    """
    Args:
        model (torch.nn.Module): Global model's architecture
        optimizer_type (str): Local optimizer.
        optimizer_args (dict): Provides necessary args for build local optimizer.
        criterion (torch.nn.CrossEntropyLoss / torch.nn.MSELoss()): Local loss function.
        epochs (int): Num of local training epoch. Personalization's local epochs may differ from others.
        batch_size (int): Batch size of training and testing.
        pers_round (int): Num of personalization round.
        cuda (bool): True for using GPUs.
        logger (fedlab.utils.Logger): Object of Logger.
    """

    def __init__(
        self,
        model,
        optimizer_type,
        optimizer_args,
        criterion,
        epochs,
        batch_size,
        pers_round,
        cuda,
        logger=Logger(),
    ):
        self.dataset = PickleDataset("femnist")
        self.optimizer_type = optimizer_type
        self.optimizer_args = optimizer_args
        self.pers_round = pers_round
        self.batch_size = batch_size
        super().__init__(
            model,
            None,
            epochs,
            get_optimizer(model, optimizer_type, optimizer_args),
            criterion,
            cuda,
            logger,
        )

    def train(self, client_id, model_parameters=None):
        """use sgd to optimize global model in client locally

        Args:
            model_parameters (torch.Tensor): serialized model parameters of global model
            client_id (int[optional]): setting specifically for multi-processing scenario
        Returns:
            [torch.Tensor, List[int, List[torch.Tensor]]]: return updated model's serialized parameters, client weight and gradients
        """
        trainloader = DataLoader(
            self.dataset.get_dataset_pickle("train", client_id), self.batch_size
        )
        if model_parameters is not None:
            SerializationTool.deserialize_model(self._model, model_parameters)
        freezed_model = deepcopy(self._model)

        self._train(client_id, self._model, trainloader, self.optimizer, self.epochs)

        gradients = []
        for old_param, new_param in zip(
            freezed_model.parameters(), self._model.parameters()
        ):
            gradients.append(old_param.data - new_param.data)
        weight = torch.tensor(len(trainloader.sampler))
        return weight, gradients

    def evaluate(self, client_id, model_parameters=None, verbose=False):
        dataset = self.dataset.get_dataset_pickle("test", client_id)
        trainset, valset = random_split(
            dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
        )
        trainloader = DataLoader(trainset, self.batch_size)
        valloader = DataLoader(valset, self.batch_size)
        if model_parameters is not None:
            SerializationTool.deserialize_model(self._model, model_parameters)
        init_loss, init_acc = evaluate(self._model, self.criterion, valloader)
        if verbose:
            self._LOGGER.info(
                f"client [{client_id}] evaluation(init)\nloss: {init_loss:.4f}\tacc: {(init_acc * 100.0):.1f}%"
            )

        # personalization
        per_model = deepcopy(self._model)
        per_optimizer = get_optimizer(
            per_model, self.optimizer_type, self.optimizer_args
        )

        self._train(client_id, per_model, trainloader, per_optimizer, self.pers_round)
        per_loss, per_acc = evaluate(per_model, self.criterion, valloader)
        if verbose:
            self._LOGGER.info(
                f"client [{client_id}] evaluation(per)\nloss: {per_loss:.4f}\tacc: {(per_acc * 100.0):.1f}%"
            )

        return (init_loss, init_acc), (per_loss, per_acc)

    def _train(self, client_id, model, dataloader, optimizer, epochs):
        model.train()
        for _ in trange(epochs, desc="client [{}]".format(client_id)):
            for x, y in dataloader:
                if self.cuda:
                    x, y = x.cuda(self.gpu), y.cuda(self.gpu)

                outputs = model(x)
                loss = self.criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

