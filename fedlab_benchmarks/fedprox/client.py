import tqdm
import torch
from copy import deepcopy

from fedlab.core.client import ClientSGDTrainer
from fedlab.utils.serialization import SerializationTool
from fedlab.utils import Logger

class FedProxTrainer(ClientSGDTrainer):
    """FedProxTrainer. 

    Details of FedProx are available in paper: https://arxiv.org/abs/1812.06127

    Args:
        model (torch.nn.Module): PyTorch model.
        data_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        epochs (int): the number of local epoch.
        optimizer (torch.optim.Optimizer, optional): optimizer for this client's model.
        criterion (torch.nn.Loss, optional): loss function used in local training process.
        cuda (bool, optional): use GPUs or not. Default: ``True``.
        logger (Logger, optional): :object of :class:`Logger`.
        mu (float): hyper-parameter of FedProx.
    """
    def __init__(self, model, data_loader, epochs, optimizer, criterion, cuda=True, logger=Logger(), mu):
        super().__init__(model, data_loader, epochs, optimizer, criterion, cuda=cuda, logger=logger)

        self.mu = mu
    
    def train(self, model_parameters):
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, model_parameters)
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for inputs, labels in tqdm(self._data_loader,
                                       desc="{}, Epoch {}".format(
                                           self._LOGGER.name, ep)):
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(
                        self.gpu)

                outputs = self._model(inputs)
                l1 = self.criterion(outputs, labels)
                l2 = 0.0

                for w0, w in zip(frz_model.parameters(),
                                 self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * self.mu * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")

        return self.model_parameters
