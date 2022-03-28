import argparse

from torch import nn


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


class MnistMLP(nn.Module):
    def __init__(self):
        super(MnistMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EmnistCNN(nn.Module):
    def __init__(self):
        super(EmnistCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(in_features=1024, out_features=62),
        )

    def forward(self, x):
        return self.net(x)


def get_model(args):
    if args.dataset == "mnist":
        if args.struct == "mlp":
            return MnistMLP()
        elif args.struct == "cnn":
            return MnistCNN()
        else:
            raise ValueError(
                'Invalid value of args.struct, expected value is "mlp" or "cnn".'
            )
    elif args.dataset == "emnist":
        if args.struct == "cnn":
            return EmnistCNN()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


if __name__ == "__main__":
    from utils import get_args
    import torch

    parser = argparse.ArgumentParser()
    args = get_args(parser)
    model = get_model(args)
    logit = model(torch.randn([5, 3, 28, 28]))
