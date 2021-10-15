import argparse
import os
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100

import sys

sys.path.append("../../../FedLab/")

from fedlab.utils.dataset import CIFAR10Partitioner, CIFAR100Partitoner
from fedlab.utils.functional import partition_report, save_dict, load_dict


def get_exp_name(args):
    exp_name = ""
    args_dict = vars(args)
    exclude_keys = ["out_dit", "data_dir"]

    for key in sorted(args_dict.keys()):
        exp_name += f"{key}_"
        if key not in exclude_keys:
            value = args_dict[key]
            if isinstance(value, float):
                exp_name += f"{value:.3f}_"
            else:
                exp_name += f"{value}_"

    return exp_name[:-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedDyn implementation: Client scale mode")

    parser.add_argument("--data-dir", type=str, default="../../../datasets/")
    parser.add_argument("--out-dir", type=str, default="./Output/")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Currently only 'cifar10' and 'cifar100' are supported")
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--partition", type=str, default="iid")
    parser.add_argument("--balance", type=bool, default=None)
    parser.add_argument("--unbalance-sgm", type=float, default=0)
    parser.add_argument("--dir-alpha", type=float, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10":
        trainset = CIFAR10(root=args.data_dir, train=True, download=True)
        partitioner = CIFAR10Partitioner
    elif args.dataset == "cifar100":
        trainset = CIFAR100(root=args.data_dir, train=True, download=True)
        partitioner = CIFAR100Partitoner
    else:
        raise ValueError(f"{args.dataset} is not supported yet.")

    partition = partitioner(targets=trainset.targets,
                            num_clients=args.num_clients,
                            balance=args.balance,
                            partition=args.partition,
                            unbalance_sgm=args.unbalance_sgm,
                            num_shards=args.num_shards,
                            dir_alpha=args.dir_alpha,
                            seed=args.seed,
                            verbose=True)
    file_name = f"{args.dataset}_{args.partition}.pkl"  # get_exp_name(args) + ".pkl"
    save_dict(partition.client_dict,
              path=os.path.join(args.out_dir, file_name))
