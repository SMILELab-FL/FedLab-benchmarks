import torch
import torchvision
import torchvision.transforms as transforms
import sys

from fedlab.utils.dataset.sampler import RawPartitionSampler

sys.path.append('../../')

from models.cnn import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST
from models.rnn import RNN_Shakespeare
from models.mlp import MLP_CelebA
from leaf.dataloader import get_LEAF_dataloader

def get_dataset(args):
    if args.dataset == 'mnist':
        root = '../../datasets/mnist/'
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(root=root,
                                              train=True,
                                              download=True,
                                              transform=train_transform)

        testset = torchvision.datasets.MNIST(root=root,
                                             train=False,
                                             download=True,
                                             transform=test_transform)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=RawPartitionSampler(trainset,
                                        client_id=args.rank,
                                        num_replicas=args.world_size - 1),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.world_size)

        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=int(
                                                     len(testset) / 10),
                                                 drop_last=False,
                                                 shuffle=False)
    elif args.dataset == 'femnist':
        trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset,
                                                      client_id=args.rank)
    elif args.dataset == 'shakespeare':
        trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset,
                                                      client_id=args.rank)
    elif args.dataset == 'celeba':
        trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset,
                                                      client_id=args.rank)
    else:
        raise ValueError("Invalid dataset:", args.dataset)

    return trainloader, testloader


def get_model(args):
    if args.dataset == "mnist":
        model = CNN_MNIST()
    elif args.dataset == 'femnist':
        model = CNN_FEMNIST()
    elif args.dataset == 'shakespeare':
        model = RNN_Shakespeare()
    elif args.dataset == 'celeba':
        model = MLP_CelebA()
    else:
        raise ValueError("Invalid dataset:", args.dataset)

    return model