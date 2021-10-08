import argparse

import sys

sys.path.append('../../../../FedLab/')

from fedlab.utils.dataset import FMNISTPartitioner
from fedlab.utils.functional import save_dict

from torchvision.datasets import FashionMNIST


# python data_partition.py --num-clients 10
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data partition')

    parser.add_argument('--num-clients', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    root = "../../../../datasets/FMNIST"
    trainset = FashionMNIST(root=root, train=True, download=True)

    # perform partition
    partition = FMNISTPartitioner(trainset.targets,
                                  num_clients=args.num_clients,
                                  partition="iid",
                                  seed=args.seed)
    save_dict(partition.client_dict, "fmnist_iid.pkl")
