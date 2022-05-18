from json import load
import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
import sys
import torch
import numpy as np
import cvxopt
torch.manual_seed(0)

from fedlab.core.client.serial_trainer import SubsetSerialTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.utils.functional import get_best_gpu, load_dict

sys.path.append("../")
from models.cnn import CNN_MNIST

def quadprog(Q, q, G, h, A, b):
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

def optim_lambdas(gradients, lambda0):
    epsilon = 0.5
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
    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    res = quadprog(Q, q, G, h, A, b)
    return res

# python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 5 --partition iid
# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=10)
parser.add_argument("--com_round", type=int, default=5)

parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)

args = parser.parse_args()

# get raw dataset
root = "../datasets/mnist/"
trainset = torchvision.datasets.MNIST(root=root,
                                      train=True,
                                      download=True,
                                      transform=transforms.ToTensor())

testset = torchvision.datasets.MNIST(root=root,
                                     train=False,
                                     download=True,
                                     transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=len(testset),
                                          drop_last=False,
                                          shuffle=False)

# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

gpu = get_best_gpu()
model = CNN_MNIST().cuda(gpu)

# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client  # client总数
data_indices = load_dict("./mnist_noniid.pkl")


# fedlab setup
local_model = deepcopy(model)

trainer = SubsetSerialTrainer(model=local_model,
                              dataset=trainset,
                              data_slices=data_indices,
                              aggregator=aggregator,
                              args={
                                  "batch_size": args.batch_size,
                                  "epochs": args.epochs,
                                  "lr": args.lr
                              })

dynamic_lambdas = np.ones(num_per_round) * 1.0 / num_per_round

# train procedure
to_select = [i for i in range(total_client_num)]
for round in range(args.com_round):
    model_parameters = SerializationTool.serialize_model(model)
    selection = random.sample(to_select, num_per_round)
    parameters = trainer.train(model_parameters=model_parameters,
                               id_list=selection,
                               aggregate=False)

    gradients = [model_parameters - model for model in parameters]
    for i, grad in enumerate(gradients):
        gradients[i] = grad / grad.norm()
    print(len(gradients))
    print(gradients[0].shape)
    # calculate lamda
    lambda0 = [1.0 / num_per_round for _ in range(num_per_round)]
    dynamic_lambdas = torch.Tensor(optim_lambdas(gradients, lambda0)).view(-1)
    dt = Aggregators.fedavg_aggregate(gradients, dynamic_lambdas)
    serialized_parameters = model_parameters - dt * args.lr
    SerializationTool.deserialize_model(model, serialized_parameters)

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    print("loss: {:.4f}, acc: {:.2f}".format(loss, acc))
