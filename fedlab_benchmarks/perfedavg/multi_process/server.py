from sys import path

path.append("../")

import argparse
from handler import PersonalizaitonHandler, FedAvgHandler, FineTuneHandler
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from server_manager import PerFedAvgSyncServerManager
from utils import get_args
from models import EmnistCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    model = EmnistCNN()
    fedavg_handler = FedAvgHandler(
        model=model,
        global_round=args.epochs,
        client_num_in_total=int(0.8 * args.client_num_in_total),
        client_num_per_round=args.client_num_per_round,
        optimizer_type="momentum_sgd",
        optimizer_args=dict(lr=args.server_lr, momentum=0.9),
        cuda=args.cuda,
        logger=Logger(log_name="fedavg"),
    )

    finetune_handler = (
        FineTuneHandler(
            model=model,
            global_round=args.fine_tune_outer_loops,
            client_num_in_total=int(0.8 * args.client_num_in_total),
            client_num_per_round=args.client_num_per_round,
            optimizer_type="sgd",
            optimizer_args=dict(lr=args.fine_tune_server_lr),
            cuda=args.cuda,
            logger=Logger(log_name="fine-tune"),
        )
        if args.fine_tune
        else None
    )

    personalization_handler = PersonalizaitonHandler(
        model=model,
        global_round=args.test_round,
        client_num_in_total=args.client_num_in_total
        - int(0.8 * args.client_num_in_total),
        client_num_per_round=args.client_num_per_round,
        cuda=args.cuda,
        logger=Logger(log_name="personalization"),
    )
    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=0,
        ethernet=args.ethernet,
    )
    manager_ = PerFedAvgSyncServerManager(
        network=network,
        fedavg_handler=fedavg_handler,
        finetune_handler=finetune_handler,
        personalization_handler=personalization_handler,
        logger=Logger(log_name="manager_server"),
    )

    manager_.run()
