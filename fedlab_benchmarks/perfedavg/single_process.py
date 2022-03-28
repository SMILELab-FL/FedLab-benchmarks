from sys import path

path.append("../")

import random
import argparse
import torch
import os
from copy import deepcopy
from fedlab.utils.functional import get_best_gpu
from fedlab.utils.logger import Logger
from fedlab.utils.serialization import SerializationTool
from trainer import PerFedAvgTrainer
from fine_tuner import LocalFineTuner
from models import get_model
from utils import get_args, get_dataloader, get_datasets, get_optimizer
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    datasets_root = (
        "../datasets/emnist" if args.dataset == "emnist" else "../datasets/mnist"
    )
    if os.path.isdir(datasets_root) is False:
        os.mkdir(datasets_root)
    datasets = get_datasets(args, datasets_root)
    logger = Logger(log_name="Personalized FedAvg")
    device = torch.device("cpu")
    if torch.cuda.is_available() and args.cuda:
        device = get_best_gpu()
    global_model = get_model(args).to(device)
    global_optimizer = get_optimizer(
        global_model, "sgd", dict(lr=args.server_lr, momentum=0.9)
    )
    criterion = torch.nn.CrossEntropyLoss()
    trainloader_list, valloader_list = get_dataloader(datasets, args)
    # seperate clients into training clients & test clients
    num_training_clients = int(0.8 * args.client_num_in_total)
    training_clients_id_list = range(num_training_clients)
    test_clients_id_list = range(num_training_clients, args.client_num_in_total)
    stats = dict(init=[], per=[])
    trainer = PerFedAvgTrainer(
        model=deepcopy(global_model),
        trainloader_list=trainloader_list,
        valloader_list=valloader_list,
        optimizer_type="sgd",
        optimizer_args=dict(lr=args.local_lr),
        criterion=criterion,
        epochs=args.inner_loops,
        pers_round=args.pers_round,
        cuda=args.cuda,
        logger=Logger(log_name="FedAvg"),
    )
    # FedAvg training
    for e in range(args.epochs):
        logger.info(f"FedAvg training epoch [{e}] ")
        selected_clients = random.sample(
            training_clients_id_list, args.client_num_per_round
        )
        all_client_weights = []
        all_client_gradients = []

        for client_id in selected_clients:
            weight, grads = trainer.train(
                client_id, SerializationTool.serialize_model(global_model)
            )
            all_client_weights.append(weight)
            all_client_gradients.append(grads)

        # FedAvg aggregation(using momentum SGD)
        global_optimizer.zero_grad()
        weights_sum = sum(all_client_weights)
        all_client_weights = [weight / weights_sum for weight in all_client_weights]
        for weight, grads in zip(all_client_weights, all_client_gradients):
            for param, grad in zip(global_model.parameters(), grads):
                if param.grad is None:
                    param.grad = torch.zeros(
                        param.size(), requires_grad=True, device=param.device
                    )
                param.grad.data.add_(grad.data * weight)
        global_optimizer.step()
        if e % 20 == 0:
            selected_clients = random.sample(
                test_clients_id_list, args.client_num_per_round
            )
            init_acc = per_acc = 0
            for client_id in selected_clients:
                init_stats, per_stats = trainer.evaluate(
                    client_id, SerializationTool.serialize_model(global_model)
                )
                init_acc += init_stats[1]
                per_acc += per_stats[1]
            stats["init"].append(init_acc / args.client_num_per_round)
            stats["per"].append(per_acc / args.client_num_per_round)

    # Plot
    if os.path.isdir("./image") is False:
        os.mkdir("./image")
    plt.plot(stats["init"])
    plt.plot(stats["per"])
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.xticks(range(len(stats["init"])), range(0, len(stats["init"]) * 20, 20))
    plt.legend(
        [
            "E = {}  init acc".format(args.epochs),
            "E = {}  pers acc".format(args.epochs),
        ]
    )
    plt.savefig(
        "./image/E={} {} clients".format(args.inner_loops, args.client_num_per_round)
    )

    # Fine-tune
    if args.fine_tune:
        server_optimizer = get_optimizer(
            global_model, "sgd", dict(lr=args.fine_tune_server_lr)
        )
        fine_tuner = LocalFineTuner(
            deepcopy(global_model),
            trainloader_list=trainloader_list,
            valloader_list=valloader_list,
            optimizer_type="adam",
            optimizer_args=dict(lr=args.fine_tune_local_lr, betas=(0, 0.999)),
            criterion=criterion,
            epochs=args.fine_tune_inner_loops,
            cuda=args.cuda,
            logger=Logger(log_name="fine-tune"),
        )
        logger.info(
            "\033[1;33mFine-tune start(epoch={})\033[0m".format(
                args.fine_tune_outer_loops
            )
        )
        for e in range(args.fine_tune_outer_loops):
            logger.info("Fine-tune epoch [{}] start".format(e))
            serialized_model_param = SerializationTool.serialize_model(global_model)
            all_clients_gradients = []
            selected_clients = random.sample(
                training_clients_id_list, args.client_num_per_round
            )
            for client_id in selected_clients:
                # send model to clients and retrieve gradients
                grads = fine_tuner.train(client_id, serialized_model_param)
                all_clients_gradients.append(grads)

            # aggregate grads and update model
            server_optimizer.zero_grad()
            for grads in all_clients_gradients:
                for param, grad in zip(global_model.parameters(), grads):
                    if param.grad is None:
                        param.grad = torch.zeros(
                            param.size(), requires_grad=True, device=param.device
                        )
                    param.grad.data.add_(grad.data.to(param.device))
            for param in global_model.parameters():
                param.grad.data.div_(len(selected_clients))
            server_optimizer.step()
        logger.info("Fine-tune end")

    # Personalization and final Evaluation
    avg_init_loss = avg_init_acc = avg_per_loss = avg_per_acc = 0
    for _ in range(args.test_round):
        init_stats = []
        per_stats = []
        selected_clients = random.sample(
            test_clients_id_list, args.client_num_per_round
        )

        for client_id in selected_clients:
            init_, per_ = trainer.evaluate(
                client_id, SerializationTool.serialize_model(global_model)
            )
            init_stats.append(init_)
            per_stats.append(per_)

        init_loss = init_acc = per_loss = per_acc = 0
        for i in range(len(selected_clients)):
            init_loss += init_stats[i][0]
            init_acc += init_stats[i][1]
            per_loss += per_stats[i][0]
            per_acc += per_stats[i][1]

        avg_init_loss += init_loss / (args.client_num_per_round * args.test_round)
        avg_init_acc += init_acc / (args.client_num_per_round * args.test_round)
        avg_per_loss += per_loss / (args.client_num_per_round * args.test_round)
        avg_per_acc += per_acc / (args.client_num_per_round * args.test_round)

    print(
        "\033[1;33m-------------------------------- RESULTS --------------------------------\033[0m"
    )
    print(
        "\033[1;33minit loss: {:.4f}\ninit acc: {:.1f}%\nper loss: {:.4f}\nper acc: {:.1f}%\033[0m".format(
            avg_init_loss, (avg_init_acc * 100.0), avg_per_loss, (avg_per_acc * 100.0)
        )
    )
