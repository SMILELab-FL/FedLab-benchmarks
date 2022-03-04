# python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 5 --partition iid

import sys

import argparse
import os
import torch
import random
from copy import deepcopy
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.utils.functional import get_best_gpu
from fedlab.utils.dataset.sampler import SubsetSampler
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

from models.cnn import CNN_MNIST
from fedprox_trainer import FedProxTrainer

parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)

parser.add_argument("--sample_ratio", type=float, default=0.1)

parser.add_argument("--batch_size", type=int, default=10)

parser.add_argument("--lr", type=float, default=0.03)

parser.add_argument("--epochs", type=int, default=5)

parser.add_argument("--partition", type=str, default="iid")

parser.add_argument("--round", type=int, default=10)

parser.add_argument(
    "--straggler", type=float, default=0.0
)  # vaild value should be in range [0, 1] and mod 0.1 == 0

parser.add_argument(
    "--optimizer", type=str, default="sgd"
)  # valid value: {"sgd", "adam", "rmsprop"}

parser.add_argument(
    "--mu", type=float, default=0.0
)  # recommended value: {0.001, 0.01, 0.1, 1.0}

args = parser.parse_args()

# get raw dataset and build corresponding dataloader
root = "../../datasets/mnist/"
trainset = datasets.MNIST(
    root=root, train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False,)

testset = datasets.MNIST(
    root=root, train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(
    testset, batch_size=len(testset), drop_last=False, shuffle=False
)

# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if torch.cuda.is_available():
    device = get_best_gpu()
else:
    device = torch.device("cpu")
model = CNN_MNIST().to(device)

# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client
criterion = nn.CrossEntropyLoss()


if args.partition == "noniid":
    data_indices = noniid_slicing(
        dataset=trainset, num_clients=total_client_num, num_shards=200
    )
else:
    data_indices = random_slicing(dataset=trainset, num_clients=total_client_num)

optimizer_dict = dict(sgd=optim.SGD, adam=optim.Adam, rmsprop=optim.RMSprop)
optimizer = optimizer_dict[args.optimizer]

# initialize training dataloaders of each client
to_select = [i for i in range(args.total_client)]
trainloader_list = [
    DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        sampler=SubsetSampler(indices=data_indices[i]),
    )
    for i in to_select
]

# train
for i in range(args.round):
    selections = random.sample(to_select, num_per_round)
    params_list = []
    client_epoch = [args.epochs] * len(selections)

    # Codes below are for realizing device heterogeneity. See paper's section 5.2 for more details
    if args.straggler > 0:
        x = random.randint(1, args.epochs)

        stragglers = (
            np.random.geometric(args.straggler, len(selections)) == 1
        )  # randomly choose client as straggler
        client_epoch = [
            x if is_straggler else args.epochs for is_straggler in stragglers
        ]

    # local train
    for c in range(len(selections)):
        local_model = deepcopy(model)
        model_param = FedProxTrainer(
            model=local_model,
            data_loader=trainloader_list[selections[c]],
            epochs=client_epoch[c],
            optimizer=optimizer(local_model.parameters(), lr=args.lr),
            criterion=criterion,
            mu=args.mu,
        ).train(SerializationTool.serialize_model(local_model))

        params_list.append(model_param)

    # update global model
    aggregated_params = aggregator(params_list)
    SerializationTool.deserialize_model(model, aggregated_params)

    # evaluate
    loss, acc = evaluate(model, criterion, test_loader)
    print(f"Epoch: {i}    loss: {loss:.4f}    accuracy: {acc:.2f}")

