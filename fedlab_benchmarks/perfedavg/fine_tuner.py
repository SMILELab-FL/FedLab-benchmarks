from sys import path

path.append("../")

import torch
from torch.utils.data import DataLoader
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger
from fedlab.utils.functional import get_best_gpu
from fedlab.core.client.trainer import ClientTrainer
from utils import get_optimizer
from tqdm import trange
from leaf.pickle_dataset import PickleDataset


class LocalFineTuner(ClientTrainer):
    """
    Args:
        model (torch.nn.Module): Global model's architecture
        optimizer_type (str): Local optimizer.
        optimizer_args (dict): Provides necessary args for build local optimizer.
        criterion (torch.nn.CrossEntropyLoss / torch.nn.MSELoss()): Local loss function.
        epochs (int): Num of local training epoch. Personalization's local epochs may differ from others.
        batch_size (int): Batch size of local training.
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
        cuda,
        logger=Logger(),
    ):
        super(LocalFineTuner, self).__init__(model, cuda)
        if torch.cuda.is_available() and cuda:
            self.device = get_best_gpu()
        else:
            self.device = torch.device("cpu")
        self.epochs = epochs
        self._criterion = criterion
        self._optimizer = get_optimizer(self._model, optimizer_type, optimizer_args)
        self._logger = logger
        self.batch_size = batch_size
        self.dataset = PickleDataset("femnist")

    def train(self, client_id, model_parameters):
        trainloader = DataLoader(
            self.dataset.get_dataset_pickle("train", client_id), self.batch_size
        )
        SerializationTool.deserialize_model(self._model, model_parameters)
        gradients = []
        for param in self._model.parameters():
            gradients.append(
                torch.zeros(param.size(), requires_grad=True, device=param.device)
            )
        for _ in trange(self.epochs, desc="client [{}]".format(client_id)):

            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                logit = self._model(x)
                loss = self._criterion(logit, y)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                for idx, param in enumerate(self._model.parameters()):
                    gradients[idx].data.add_(param.grad.data)
        return gradients
