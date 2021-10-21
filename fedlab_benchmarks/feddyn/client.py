import torch

import os
import numpy as np
import sys

sys.path.append("../../../FedLab/")

from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger

from config import local_grad_vector_file_pattern, clnt_params_file_pattern


class FedDynSerialTrainer(SubsetSerialTrainer):
    def __init__(self, model,
                 dataset,
                 data_slices,
                 client_weights=None,
                 rank=None,
                 logger=Logger(),
                 cuda=True,
                 args=None):
        super().__init__(model,
                         dataset,
                         data_slices,
                         aggregator=None,
                         logger=logger,
                         cuda=cuda,
                         args=args)
        self.client_weights = client_weights
        self.rank = rank
        self.round = 0  # global round, try not to use package to inform global round

    def _train_alone(self,
                     cld_mdl_params,
                     avg_mdl_param,
                     local_grad_vector,
                     train_loader,
                     client_id,
                     lr,
                     alpha_coef,
                     epochs):
        """
        cld_mdl_params: serialized model params of cloud model (server model)
        train_loader:
        client_id:
        lr:
        alpha_coef:
        epochs:
        avg_mdl_param:  model avg of selected clients from last round
        local_grad_vector:
        """
        weight_decay = self.args['weight_decay']
        max_norm = self.args['max_norm']
        print_freq = self.args['print_freq']

        SerializationTool.deserialize_model(self._model, cld_mdl_params)  # load model params
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(),
                                    lr=lr,
                                    weight_decay=alpha_coef + weight_decay)
        self._model.train()

        for e in range(epochs):
            # Training
            epoch_loss = 0
            for imgs, targets in train_loader:
                if self.cuda:
                    imgs, targets = imgs.cuda(self.gpu), targets.cuda(self.gpu)

                y_pred = self.model(imgs)

                # Get f_i estimate
                # equal to CrossEntropyLoss(reduction='sum') / targets.shape[0]
                loss_f_i = loss_fn(y_pred, targets.long())

                # Get linear penalty on the current parameter estimates
                # Note: DO NOT use SerializationTool.serialize_model() to serialize model params
                # here, they get same numeric result but result from SerializationTool doesn't
                # have 'grad_fn=<CatBackward>' !!!!!
                local_par_list = None
                for param in self.model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                # print(f"local_par_list.device={local_par_list.device}")
                # print(f"avg_mdl_param.device={avg_mdl_param.device}")
                # print(f"local_grad_vector.device={local_grad_vector.device}")
                loss_algo = alpha_coef * torch.sum(
                    local_par_list * (-avg_mdl_param + local_grad_vector))
                loss = loss_f_i + loss_algo

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=max_norm)  # Clip gradients
                optimizer.step()
                epoch_loss += loss.item() * targets.shape[0]

            if (e + 1) % print_freq == 0:
                epoch_loss /= len(self.data_slices[client_id])
                if weight_decay is not None:
                    # Add L2 loss to complete f_i
                    serialized_params = SerializationTool.serialize_model(self.model).numpy()
                    epoch_loss += (alpha_coef + weight_decay) / 2 * np.dot(serialized_params,
                                                                           serialized_params)
                self._LOGGER.info(
                    f"Round {self.round + 1}: Client {client_id:3d}, Epoch {e + 1}/{epochs}, Training Loss: {epoch_loss:.4f}")


    def train(self, model_parameters, id_list, aggregate=False):
        param_list = []
        orig_lr = self.args['lr']
        lr_decay_per_round = self.args['lr_decay_per_round']
        lr = orig_lr * (lr_decay_per_round ** self.round)  # using learning rate decay
        epochs = self.args['epochs']

        self._LOGGER.info(
            f"Round {self.round + 1}: Local training with client id list: {id_list}")
        for cid in id_list:
            self._LOGGER.info(
                f"Round {self.round + 1}: Starting training procedure of client [{cid}]")

            data_loader = self._get_dataloader(client_id=cid)
            # read local grad vector from files
            global_cid = self._local_to_global_map(cid)
            local_grad_vector_file = os.path.join(self.args['out_dir'], local_grad_vector_file_pattern.format(cid=global_cid))
            local_grad_vector = torch.load(local_grad_vector_file)  # TODO: need check here
            alpha_coef_adpt = self.args['alpha_coef'] / self.client_weights[cid]

            self._train_alone(cld_mdl_params=model_parameters,
                              avg_mdl_param=model_parameters.data.cuda(self.gpu),
                              # can only be *.data !!
                              local_grad_vector=local_grad_vector.data.cuda(self.gpu),
                              train_loader=data_loader,
                              client_id=cid,
                              alpha_coef=alpha_coef_adpt,
                              lr=lr,
                              epochs=epochs)
            param_list.append(self.model_parameters)
            # save serialized params of current client into file
            clnt_params_file = os.path.join(self.args['out_dir'], clnt_params_file_pattern.format(cid=global_cid))
            torch.save(self.model_parameters, clnt_params_file)
            self._LOGGER.info(f"Round {self.round + 1}: Client {cid:3d} serialized params save to {clnt_params_file}")

            # update local gradient vector of current client, and save to file
            # print(f"local_grad_vector.device={local_grad_vector.device}")
            # print(f"self.model_parameters.device={self.model_parameters.device}")
            # print(f"model_parameters.device={model_parameters.device}")
            local_grad_vector += self.model_parameters - model_parameters
            torch.save(local_grad_vector, local_grad_vector_file)
            self._LOGGER.info(f"Round {self.round + 1}: Client {cid:3d} serialized gradients save to {local_grad_vector_file}")

            self._LOGGER.info(f"Round {self.round + 1}: Client {cid:3d} DONE")

        self._LOGGER.info(f"Round {self.round + 1}: Serial Trainer DONE")
        self.round += 1  # trainer global round counter update

        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list)
            return aggregated_parameters
        else:
            return param_list

    def _local_to_global_map(self, local_client_id, client_num_per_rank=10):
        """
        NOTE: this function can only be used for simulations where each trainer has same number of
        clients!!!

        Args:
            local_client_id: local client id on current client process
            client_num_per_rank: number of clients on each client process

        Returns:

        """
        global_client_id = (self.rank - 1) * client_num_per_rank + local_client_id
        return global_client_id
