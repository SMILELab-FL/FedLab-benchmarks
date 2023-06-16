import os
import argparse
import time
import sys
import torch


sys.path.append("../../")

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from fedlab.utils import SerializationTool, Aggregators, Logger
from fedlab.utils.functional import evaluate
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab_benchmarks.models.cnn import CNN_CIFAR10, CNN_FEMNIST
from fedlab_benchmarks.models.mlp import MLP
from fedlab_benchmarks.cfl.cfl_trainer import CFLTrainer
from fedlab_benchmarks.leaf.dataloader import get_LEAF_all_test_dataloader
from fedlab_benchmarks.cfl.helper import ExperimentLogger, display_train_stats


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


class StandaloneServer(ModelMaintainer):
    def __init__(self, model, cuda, client_trainer, args) -> None:
        super().__init__(model, cuda)
        self.client_trainer = client_trainer
        self.client_num = self.client_trainer.client_num
        self.client_id = [i for i in range(self.client_num)]
        self.find_cluster_content = {
            0: np.arange(self.client_num)}  # find_cluster_content[cluster_id] -> [id_1, id_2, ...]
        self.find_client_cluster = np.zeros(self.client_num)  # find_client_cluster[cid] -> cluster id
        self.cluster_parameters = {0: self.model_parameters}
        serial_model_size = self.model_parameters.size()
        self.clients_updates = [torch.zeros((serial_model_size)) for _ in range(self.client_num)]
        self.clients_dWs = [torch.zeros((serial_model_size)) for _ in range(self.client_num)]

        self.weights = self.client_trainer.weights

        self.args = args
        self.EPS_1 = args.EPS_1
        self.EPS_2 = args.EPS_2

    def main(self):
        accuracy = []
        for rd in range(self.args.com_round):
            # all clients join
            for cluster_id, cluster_param in self.cluster_parameters.items():
                cluster_clients = self.find_cluster_content[cluster_id]
                updates, dWs = self.client_trainer.local_process(cluster_clients, cluster_param)
                for id, update, dW in zip(cluster_clients, updates, dWs):
                    self.clients_updates[id] = update.data
                    self.clients_dWs[id] = dW.data

            similarities = self.dt_matrix(self.clients_dWs)
            find_cluster_content_new = {}
            count = 0
            for cluster_id, idc in self.find_cluster_content.items():
                max_norm, mean_norm = self.compute_max_mean_update_norm([self.clients_dWs[i] for i in idc])
                if mean_norm < self.EPS_1 and max_norm > self.EPS_2 and len(idc) > 2 and rd > args.warm_round:
                    c1, c2 = self.cluster_clients(similarities[idc][:, idc])
                    c1, c2 = idc[c1], idc[c2]  # index transfer

                    find_cluster_content_new[count] = c1
                    find_cluster_content_new[count + 1] = c2
                    count += 2

                    self.args.exp_logger.info(f'split: {rd}, cluster: {find_cluster_content_new}')
                    self.args.cfl_stats.log({"split": rd})
                else:
                    find_cluster_content_new[count] = idc
                    count += 1

            args.exp_logger.info(
                "Round:{}, cluster_indices: {}, mean_norm: {:.4f}, max_norm:{:.4f}".format(rd,
                                                                                           find_cluster_content_new,
                                                                                           mean_norm, max_norm))
            self.find_cluster_content = find_cluster_content_new
            # update client to cluster map, and aggregate cluster model by cluster
            for cluster_id, idc in self.find_cluster_content.items():
                for id in idc:
                    self.find_client_cluster[id] = cluster_id
                self.cluster_parameters[cluster_id] = Aggregators.fedavg_aggregate(
                    [self.clients_updates[i] for i in idc],
                    weights=[self.weights[i] for i in idc])

            # test all clusters
            if (rd + 1) % 1 == 0:
                local_acc = []
                acc_clusters = []
                for cluster_id, idc in self.find_cluster_content.items():
                    if self.args.dataset == 'femnist':
                        cluster_testset = ConcatDataset(
                            [self.client_trainer.dataset.get_dataset_pickle("test", id) for id in idc])
                        dsize = len(cluster_testset)
                        cluster_testloader = DataLoader(cluster_testset, batch_size=4096)

                    else:
                        # use the first id to get testdata
                        dsize, cluster_testloader = self.client_trainer.get_testloader(idc[0])

                    SerializationTool.deserialize_model(self._model, self.cluster_parameters[cluster_id])

                    criterion = torch.nn.CrossEntropyLoss()
                    _, acc = evaluate(model, criterion, cluster_testloader)
                    local_acc.append((acc, dsize))
                    acc_clusters.append(acc)
                    args.exp_logger.info("Round {}, Cluster {} - Global Test acc: {:.4f}".format(rd, idc, acc))

                weighted_acc = 0
                all_size = 0
                for acc, dsize in local_acc:
                    weighted_acc += acc * dsize
                    all_size += dsize
                weighted_acc /= all_size
                accuracy.append(weighted_acc)
                args.exp_logger.info("Round: {}, weighted acc: {}".format(rd, weighted_acc))

                self.args.cfl_stats.log({"acc_clusters": acc_clusters, "mean_norm": mean_norm, "max_norm": max_norm,
                                         "rounds": rd, "clusters": [list(x) for x in self.find_cluster_content.values()]})

                display_train_stats(self.args.cfl_stats, self.args.EPS_1, self.args.EPS_2, self.args.com_round)

    def dt_matrix(self, grad_list):
        a = b = torch.stack(grad_list, dim=0)
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        res = torch.mm(a_norm, b_norm.transpose(0, 1))
        return res

    def cluster_clients(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten()
        c2 = np.argwhere(clustering.labels_ == 1).flatten()
        return c1, c2

    def compute_max_mean_update_norm(self, grad_list):
        a = torch.stack(grad_list, dim=0)
        return torch.max(a.norm(dim=1)).item(), torch.norm(torch.mean(a, dim=0)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone training example")
    # server
    parser.add_argument("--n", type=int, default=20)  # the number of clients, femnist: 3597, mnist: 400
    parser.add_argument("--com_round", type=int, default=100)

    # trainer
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--EPS_1", type=float, default=0.5)
    parser.add_argument("--EPS_2", type=float, default=1.0)
    parser.add_argument("--warm_round", type=int, default=20)

    parser.add_argument("--dataset", type=str, default="emnist")
    parser.add_argument("--augment", type=str, default="rotated")
    parser.add_argument("--process_data", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.root = "../datasets/{}/".format(args.dataset)
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
    args.save_dir = "./datasets/{}_{}_seed{}_{}/".format(args.augment, args.dataset, args.seed, args.n)

    print("Load model and data")
    if args.dataset == "femnist":
        model = CNN_FEMNIST()
        args.save_dir = '../leaf/pickle_datasets/'  # pickle dataset root
        args.test_loader = get_LEAF_all_test_dataloader("femnist", 4096, args.root, args.save_dir)
    elif args.dataset == "mnist":
        model = MLP(input_size=28*28, output_size=10)
    elif args.dataset == "cifar10":
        model = CNN_CIFAR10()
    elif args.dataset == "emnist":
        model = CNN_FEMNIST()

    trainer = CFLTrainer(model, args, cuda=True)

    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    args.time_stamp = time.strftime('%m-%d-%H:%M', time.localtime())
    dir = "./logs/runs-CFL-{}-{}-n{}-EPS1_{}_EPS2_{}-seed{}-time-{}".format(args.augment, args.dataset, args.n, args.EPS_1,
                                                                         args.EPS_2, args.seed, args.time_stamp)
    os.mkdir(dir)
    args.dir = dir
    args.exp_logger = Logger("CFL", "{}/{}.log".format(dir, args.dataset))
    args.exp_logger.info(str(args))
    args.cfl_stats = ExperimentLogger()

    # cfl training
    print("CFL training")
    server = StandaloneServer(model, True, trainer, args)
    server.main()
