import torch
from copy import deepcopy
from fedlab.utils.serialization import SerializationTool
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.utils.functional import evaluate
from fedlab.utils.logger import Logger
from torch import nn, optim
from utils import get_optimizer
from tqdm import trange


class PerFedAvgTrainer(ClientSGDTrainer):
    """
    Args:
        model (torch.nn.Module): Global model's architecture
        trainloader_list (List[torch.utils.data.DataLoader]): Consider as all client's local train dataloader.
        valloader_list (List[torch.utils.data.DataLoader]): Consider as all client's local val dataloader.
        optimizer_type (str): Local optimizer.
        optimizer_args (dict): Provides necessary args for build local optimizer.
        criterion (torch.nn.CrossEntropyLoss / torch.nn.MSELoss()): Local loss function.
        epochs (int): Num of local training epoch. Personalization's local epochs may differ from others.
        cuda (bool): True for using GPUs.
        logger (fedlab.utils.Logger): Object of Logger.
    """

    def __init__(
        self,
        model,
        trainloader_list,
        valloader_list,
        optimizer_type,
        optimizer_args,
        criterion,
        epochs,
        pers_round,
        cuda,
        logger=Logger(),
    ):
        self.trainloader_list = trainloader_list
        self.valloader_list = valloader_list
        self.optimizer_type = optimizer_type
        self.optimizer_args = optimizer_args
        self.pers_round = pers_round
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
        trainloader = self.trainloader_list[client_id]
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
        trainloader = self.trainloader_list[client_id]
        valloader = self.valloader_list[client_id]
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


if __name__ == "__main__":
    # For testing only. Actual main() is in single_process.py
    from utils import get_args, get_dataloader, get_datasets
    from models import get_model
    import argparse

    parser = argparse.ArgumentParser()
    args = get_args(parser)
    dataset = get_datasets(args)
    train, val, test = get_dataloader(dataset, args)
    model = get_model(args)
    optimzier = optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    trainer = PerFedAvgTrainer(
        model,
        train,
        val,
        "sgd",
        dict(lr=1e-2),
        criterion,
        10,
        False,
        Logger(log_name="node 0"),
    )
    weight, gradients = trainer.train(0, SerializationTool.serialize_model(model))
    trainer.evaluate(0)
