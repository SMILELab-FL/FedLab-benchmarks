import argparse
from copy import deepcopy
import cvxopt
import numpy as np
import torch
import torchvision
from torchvision import transforms

from fedlab.utils.logger import Logger
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.network import DistNetwork

from fedlab.utils import Aggregators, SerializationTool, Logger
from fedlab.utils.functional import evaluate
from setting import get_model, get_dataloader


class FedMGDA_handler(SyncParameterServerHandler):
    """Refer to GitHub implementation https://github.com/WwZzz/easyFL """
    def __init__(self,
                 model,
                 global_round,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=Logger()):
        
        super().__init__(model, global_round, sample_ratio, cuda, logger)

        self.epsilon = 0.5  # 0 for fedavg, 1 for fedmdga
        self.learning_rate = 0.1  # global lr
        self.gradients = []

        testset = torchvision.datasets.MNIST(root='../datasets/mnist/',
                                             train=False,
                                             download=True,
                                             transform=transforms.ToTensor())
        self.testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=int(
                                                     len(testset) / 10),
                                                 drop_last=False,
                                                 shuffle=False)
        
        
    def _update_global_model(self, payload):
        self.gradients += payload
        self.dynamic_lambdas = np.ones(
            self.client_num_per_round) * 1.0 / self.client_num_per_round

        if len(self.gradients) == self.client_num_per_round:
            self.aggregate()
            self.gradients = []
            self.round += 1
            return True

    def aggregate(self):
        gradients = self.gradients
        # clip gradients
        
        for i, grad in enumerate(gradients):
            gradients[i] = grad / grad.norm()
        
        # calculate lamda
        lambda0 = [
            1.0 / self.client_num_per_round
            for _ in range(self.client_num_per_round)
        ]
        # optimize lambdas
        self.dynamic_lambdas = torch.Tensor(self.optim_lambdas(gradients, lambda0)).view(-1)
        self._LOGGER.info("lambdas {}".format(self.dynamic_lambdas))
        # aggregate grads
        dt = Aggregators.fedavg_aggregate(gradients, self.dynamic_lambdas)
        self._LOGGER.info("dt {}".format(dt.norm()))
        serialized_parameters = self.model_parameters - self.learning_rate * dt
        SerializationTool.deserialize_model(self._model, serialized_parameters)

        loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), self.testloader)
        self._LOGGER.info("evaluate loss {}, acc {}".format(loss, acc))

    def optim_lambdas(self, gradients, lambda0):
        n = len(gradients)
        J_t = [grad.numpy() for grad in gradients]
        J_t = np.array(J_t)
        # target function
        Q = 2 * np.dot(J_t, J_t.T)
        q = np.array([[0] for i in range(n)])
        # equality constrint
        A = np.ones(n).T
        b = np.array([1])
        # boundary
        lb = np.array([max(0, lambda0[i] - self.epsilon) for i in range(n)])
        ub = np.array([min(1, lambda0[i] + self.epsilon) for i in range(n)])
        G = np.zeros((2 * n, n))
        for i in range(n):
            G[i][i] = -1
            G[n + i][i] = 1
        h = np.zeros((2 * n, 1))
        for i in range(n):
            h[i] = -lb[i]
            h[n + i] = ub[i]
        res = self.quadprog(Q, q, G, h, A, b)
        return res

    def quadprog(self, Q, q, G, h, A, b):
        """
        Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
        Output: Numpy array of the solution
        """
        Q = cvxopt.matrix(Q.tolist())
        q = cvxopt.matrix(q.tolist(), tc='d')
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b.tolist(), tc='d')
        sol = cvxopt.solvers.qp(Q, q.T, G.T, h.T, A.T, b)
        return np.array(sol['x'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=0.1)

    args = parser.parse_args()

    model = get_model(args)
    LOGGER = Logger(log_name="server", log_file="./server.log")
    handler = FedMGDA_handler(model,
                              global_round=args.round,
                              logger=LOGGER,
                              sample_ratio=args.sample)
    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0,
                          ethernet=args.ethernet)
    manager_ = SynchronousServerManager(handler=handler,
                                        network=network,
                                        logger=LOGGER)
    manager_.run()
