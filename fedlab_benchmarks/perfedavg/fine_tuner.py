import torch
import argparse
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger
from fedlab.utils.functional import get_best_gpu, evaluate
from fedlab.core.client.trainer import ClientTrainer
from utils import get_optimizer, get_datasets, get_dataloader, get_args
from models import get_model
from tqdm import trange


class LocalFineTuner(ClientTrainer):
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
        self.trainloader_list = trainloader_list
        self.valloader_list = valloader_list
        self._optimizer = get_optimizer(self._model, optimizer_type, optimizer_args)
        self._logger = logger

    def train(self, client_id, model_parameters, validation=False):
        trainloader = self.trainloader_list[client_id]
        valloader = self.valloader_list[client_id]
        if validation:
            print(f"client [{client_id}]: evaluation(before training):")
            loss, acc = evaluate(self._model, self._criterion, valloader)
            self._logger.info(
                "client [{}]\nloss: {:.4f}\naccuracy: {:.1f}%".format(
                    client_id, loss, acc
                )
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

        if validation:
            print("client [{}]: evaluation(after training):".format(client_id))
            loss, acc = evaluate(self._model, self._criterion, valloader)
            self._logger.info(
                "client [{}]\nloss: {:.4f}\naccuracy: {:.1f}%".format(
                    client_id, loss, acc
                )
            )
        return gradients


if __name__ == "__main__":
    # For testing only. Actual main() is in single_process.py
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    net = get_model(args)
    dataset = get_datasets(args)
    trainloader_list, valloader_list, _ = get_dataloader(dataset, args)
    trainer = LocalFineTuner(
        net,
        trainloader_list,
        valloader_list,
        get_optimizer(net, "adam", dict(lr=args.fine_tune_local_lr, betas=(0, 0.999))),
        torch.nn.CrossEntropyLoss(),
        10,
        False,
    )
    grads = trainer.train(0, SerializationTool.serialize_model(net), validation=True)
