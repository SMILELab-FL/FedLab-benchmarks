from torch.optim import Adam, SGD


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
        "--client_num_in_total", type=int, default=3597, help="total num of clients",
    )
    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=20,
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
        "--inner_loops", type=int, default=10, help="local epochs in FedAvg section"
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
