import argparse
import torch
import os
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fedlab.utils.dataset.sampler import SubsetSampler
from fedlab.utils.functional import evaluate


def list_dir(root):
    dir_list = [os.path.join(root, d) for d in os.listdir(root)]
    return dir_list


def get_datasets(args, datasets_root):
    trainset = None
    testset = None
    if args.dataset == "mnist":
        train_root = test_root = datasets_root
        trainset = datasets.MNIST(
            root=train_root, train=True, transform=transforms.ToTensor(), download=True
        )
        testset = datasets.MNIST(
            root=test_root, train=False, transform=transforms.ToTensor(), download=True
        )
    elif args.dataset == "emnist":
        train_root = test_root = datasets_root
        trainset = datasets.EMNIST(
            root=train_root,
            split="byclass",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )
        testset = datasets.EMNIST(
            root=test_root,
            split="byclass",
            train=False,
            transform=transforms.ToTensor(),
            download=True,
        )
    return trainset, testset


def get_dataloader(datasets, args):
    """generate torch.utils.data.DataLoader of train, val, testset

    Args:
        datasets (torchvision.datasets): origin trainset and testset
        args (Namespace): provides necessary args for guiding generation

    Returns:
        tuple[List[DataLoader], List[DataLoader]]: tranloader_list, valloader_list
        trainloader_list[i] and valloader_list[i] are for client i specifically.
    """
    print(
        "Generating client's train dataloader and test dataloader, it may takes a while, please be patient. :-)"
    )
    trainset, testset = datasets

    train_client_num = int(0.8 * args.client_num_in_total)
    test_client_num = args.client_num_in_total - train_client_num

    # Non-IID
    train_sample_indices = _noniid_slicing(
        trainset, train_client_num, 2 * train_client_num
    )
    val_sample_indices = _noniid_slicing(testset, test_client_num, 2 * test_client_num)

    # IID
    # random.seed(1000)
    # num_items = int(len(trainset) / train_client_num)
    # train_sample_indices, all_idxs = {}, [i for i in range(len(trainset))]
    # for i in range(train_client_num):
    #     train_sample_indices[i] = random.sample(all_idxs, num_items)
    #     all_idxs = list(set(all_idxs) - set(train_sample_indices[i]))
    # num_items = int(len(testset) / test_client_num)
    # val_sample_indices, all_idxs = {}, [i for i in range(len(testset))]
    # for i in range(test_client_num):
    #     val_sample_indices[i] = random.sample(all_idxs, num_items)
    #     all_idxs = list(set(all_idxs) - set(val_sample_indices[i]))

    train_part_indices = {}
    val_part_indices = {}
    for client_id, indices in train_sample_indices.items():
        train_part_indices.update({client_id: indices[: int(0.8 * len(indices))]})
        val_part_indices.update({client_id: indices[int(0.8 * len(indices)) :]})
    for client_id, indices in val_sample_indices.items():
        train_part_indices.update(
            {client_id + train_client_num: indices[: int(0.8 * len(indices))]}
        )
        val_part_indices.update(
            {client_id + train_client_num: indices[int(0.8 * len(indices)) :]}
        )

    trainloader_list = [
        DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            sampler=SubsetSampler(train_part_indices[idx]),
        )
        for idx in range(train_client_num)
    ] + [
        DataLoader(
            dataset=testset,
            batch_size=args.batch_size,
            sampler=SubsetSampler(train_part_indices[idx]),
        )
        for idx in range(train_client_num, args.client_num_in_total)
    ]

    valloader_list = [
        DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            sampler=SubsetSampler(val_part_indices[idx]),
        )
        for idx in range(train_client_num)
    ] + [
        DataLoader(
            dataset=testset,
            batch_size=args.batch_size,
            sampler=SubsetSampler(val_part_indices[idx]),
        )
        for idx in range(train_client_num, args.client_num_in_total)
    ]
    return trainloader_list, valloader_list


def _noniid_slicing(dataset, num_clients, num_shards):
    """Slice a dataset for non-IID. Same code from fedlab.utils.dataset.slicing.noniid_slicing(), additionally with fixed random seed.
    Args:
        dataset (torch.utils.data.Dataset): Dataset to slice.
        num_clients (int):  Number of client.
        num_shards (int): Number of shards.
    
    Notes:
        The size of a shard equals to ``int(len(dataset)/num_shards)``.
        Each client will get ``int(num_shards/num_clients)`` shards.

    Returns：
        dict: ``{ 0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset }``
    """
    # Designated random seed to make sure all workers split datasets in the same way.
    np.random.seed(1000)

    total_sample_nums = len(dataset)
    size_of_shards = int(total_sample_nums / num_shards)

    # the number of shards that each one of clients can get
    shard_pc = int(num_shards / num_clients)

    dict_users = {i: np.array([], dtype="int64") for i in range(num_clients)}

    labels = np.array(dataset.targets)
    idxs = np.arange(total_sample_nums)

    # sort sample indices according to labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[
        0, :
    ]  # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]

    # assign
    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shard_pc, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (
                    dict_users[i],
                    idxs[rand * size_of_shards : (rand + 1) * size_of_shards],
                ),
                axis=0,
            )

    return dict_users


def get_optimizer(model, optimizer, args=None):
    if args is None:
        args = {}
    if optimizer == "sgd":
        _lr = 2e-2 if "lr" not in args.keys() else args["lr"]
        return SGD(model.parameters(), lr=_lr)
    elif optimizer == "momentum_sgd":
        _lr = 1e-2 if "lr" not in args.keys() else args["lr"]
        _momentum = 0.9 if "momentum" not in args.keys() else args["momentum"]
        return SGD(model.parameters(), lr=_lr, momentum=_momentum)
    elif optimizer == "adam":
        _lr = 1e-3 if "lr" not in args.keys() else args["lr"]
        _betas = (0.9, 0.999) if "betas" not in args.keys() else args["betas"]
        return Adam(model.parameters(), lr=_lr, betas=_betas)
    raise NotImplementedError


def get_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        default="emnist",
        help="dataset name, expected of emnist or mnist",
    )
    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=3400,
        help="total num of clients, default value is set according to paper",
    )
    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=5,
        help="determine how many clients join training in one communication round",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="batch size of local training in FedAvg and fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=500, help="communication round")
    parser.add_argument(
        "--inner_loops", type=int, default=20, help="local epochs in FedAvg section"
    )
    parser.add_argument(
        "--server_lr",
        type=float,
        default=1.0,
        help="server optimizer lr in FedAvg section",
    )
    parser.add_argument(
        "--local_lr",
        type=float,
        default=2e-2,
        help="local optimizer lr in FedAvg section",
    )
    parser.add_argument(
        "--fine_tune",
        type=bool,
        default=True,
        help="determine whether perform fine-tune",
    )
    parser.add_argument(
        "--fine_tune_outer_loops",
        type=int,
        default=100,
        help="outer epochs in fine-tune section",
    )
    parser.add_argument(
        "--fine_tune_inner_loops",
        type=int,
        default=10,
        help="inner epochs in fine-tune section",
    )
    parser.add_argument(
        "--fine_tune_server_lr",
        type=float,
        default=1e-2,
        help="server optimizer lr in fine-tune section",
    )
    parser.add_argument(
        "--fine_tune_local_lr",
        type=float,
        default=1e-3,
        help="local optimizer lr in fine-tune section",
    )
    parser.add_argument(
        "--test_round", type=int, default=100, help="num of round of final testing"
    )
    parser.add_argument(
        "--pers_round", type=int, default=5, help="num of round of personalization"
    )
    parser.add_argument(
        "--cuda",
        type=bool,
        default=True,
        help="True for using GPUS, False for using CPU",
    )
    parser.add_argument(
        "--struct",
        type=str,
        default="cnn",
        help="architecture of model, expected of mlp or cnn",
    )
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3002")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--ethernet", type=str, default=None)
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    # For testing only. Actual main() is in single_process.py
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    dataset = get_datasets(args)
    train, val = get_dataloader(dataset, args)
    from models import get_model

    model = get_model(args)
    loader = train[0]
    criterion = torch.nn.CrossEntropyLoss()
    optimzier = torch.optim.SGD(model.parameters(), lr=1e-2)

    for x, y in loader:
        logit = model(x)
        loss = criterion(logit, y)

        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

    loss, avg = evaluate(model, criterion, val[0])
