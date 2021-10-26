"""
Standalone mode for FedDyn, no server handler and no network manager.
Only client trainer is needed.
"""

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

import argparse
import os
import random
from copy import deepcopy
from pathlib import Path
import sys

sys.path.append("../../../FedLab/")

from fedlab.utils.logger import Logger
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import get_best_gpu, save_dict, load_dict

import models
from utils import evaluate
from config import cifar10_config, balance_iid_data_config, debug_config
from client import FedDynSerialTrainer, FedDynSerialTrainer_v2, FedAvgSerialTrainer


def write_file(accs, losses, config):
    file_name = os.path.join(config['out_dir'],
                             f"{config['model_name']}_{config['partition']}_{config['dataset']}.txt")
    record = open(file_name, "w")

    record.write(str(config) + "\n")
    record.write(f"acc:" + str(accs) + "\n")
    record.write(f"loss:" + str(losses) + "\n")
    record.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FedDyn-standalone demo in FedLab")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--alg", type=str, default='FedDyn')
    parser.add_argument("--partition", type=str, default='iid', help="Choose from ['iid', 'niid']")
    parser.add_argument("--data-dir", type=str, default='../../../datasets')
    parser.add_argument("--out-dir", type=str, default='./Output/')
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # get basic config
    if args.partition == 'iid':
        alg_config = cifar10_config
        data_config = balance_iid_data_config

    # get basic model
    gpu = get_best_gpu()
    model_name = alg_config['model_name']
    server_model = getattr(models, model_name)(model_name).cuda(gpu)
    client_model = deepcopy(server_model)
    num_clients = args.num_clients
    num_per_round = int(num_clients * args.sample_ratio)
    aggregator = Aggregators.fedavg_aggregate

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if args.partition == 'iid':
        data_indices = load_dict(os.path.join('./Output', "cifar10_iid.pkl"))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'),
                                            train=True,
                                            download=False,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR10'),
                                           train=False,
                                           download=False,
                                           transform=transform_test)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=alg_config['test_batch_size'],
                                              drop_last=False,
                                              shuffle=False)

    # ====== init weight list for each client
    total_sample_num = len(trainset)
    weight_list = {
        cid: len(data_indices[cid]) / total_sample_num for cid in range(num_clients)
    }
    if args.alg == 'FedDyn':
        weight_list = {key: value * num_clients for key, value in weight_list.items()}

    trainer_logger = Logger(f"StandaloneClientTrainer",
                            os.path.join(args.out_dir, f"ClientTrainer.txt"))
    server_logger = Logger("Server",
                           os.path.join(args.out_dir, "Server.txt"))

    alg_config['out_dir'] = args.out_dir
    if args.alg == 'FedDyn':
        trainer = FedDynSerialTrainer_v2(model=client_model,
                                         dataset=trainset,
                                         data_slices=data_indices,
                                         client_weights=weight_list,
                                         rank=args.rank,
                                         logger=trainer_logger,
                                         args=alg_config)
    elif args.alg == 'FedAvg':
        trainer = FedAvgSerialTrainer(model=client_model,
                                      dataset=trainset,
                                      data_slices=data_indices,
                                      client_weights=weight_list,
                                      rank=args.rank,
                                      logger=trainer_logger,
                                      args=alg_config)
    else:
        raise ValueError(f"args.alg={args.alg} is not supported.")

    test_acc_hist = []
    test_loss_hist = []
    init_params = SerializationTool.serialize_model(server_model)
    clnt_params_list = [init_params.data for _ in range(num_clients)]

    for r in range(alg_config['round']):
        model_params = SerializationTool.serialize_model(server_model)
        selected_cid = sorted(random.sample(range(num_clients), num_per_round))
        if args.alg == 'FedDyn':
            params_list, local_grad_vector_list = trainer.train(model_parameters=model_params,
                                                                id_list=selected_cid,
                                                                aggregate=False)
            server_logger.info(
                "Model parameters aggregation, number of aggregation elements {}".format(
                    len(params_list)))

            for idx, cid in enumerate(selected_cid):
                clnt_params_list[cid] = params_list[idx].data

            avg_mdl_param = Aggregators.fedavg_aggregate(params_list)
            avg_local_grad = Aggregators.fedavg_aggregate(local_grad_vector_list)
            cld_mdl_param = avg_mdl_param + avg_local_grad
            SerializationTool.deserialize_model(server_model, cld_mdl_param)
            server_logger.info("Server model update DONE")

            # avg_model = getattr(models, model_name)(model_name)
            # SerializationTool.deserialize_model(avg_model, avg_mdl_param)
            #
            # all_model = getattr(models, model_name)(model_name)
            # all_model_params = Aggregators.fedavg_aggregate(clnt_params_list)
            # SerializationTool.deserialize_model(all_model, all_model_params)

        else:
            # FedAvg
            params_list = trainer.train(model_parameters=model_params,
                                        id_list=selected_cid,
                                        aggregate=False)
            server_logger.info(
                "Model parameters aggregation, number of aggregation elements {}".format(
                    len(params_list)))
            curr_weight_sum = sum([weight_list[cid] for cid in selected_cid])
            serialized_parameters = Aggregators.fedavg_aggregate(
                params_list) * num_per_round / curr_weight_sum
            SerializationTool.deserialize_model(server_model, serialized_parameters)

        # server model evaluation
        test_loss, test_acc = evaluate(server_model, nn.CrossEntropyLoss(), test_loader)
        test_acc_hist.append(test_acc)
        test_loss_hist.append(test_loss)
        write_file(test_acc_hist, test_loss_hist, alg_config)
