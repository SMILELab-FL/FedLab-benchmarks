import os
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from fedlab.utils.dataset.slicing import random_slicing


class BaseDataset(data_utils.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class RotatedPartitioner:
    def __init__(self, root, save_dir, dataset_name):
        self.root = os.path.expanduser(root)
        self.dir = save_dir
        self.dataset_name = dataset_name
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

    def pre_process(self, thetas=[0, 90, 180, 270], shards=100):
        # train
        if self.dataset_name == 'mnist':
            pre_data = datasets.MNIST(self.root, train=True)
            data_transform = transforms.ToTensor()
        elif self.dataset_name == 'cifar10':
            pre_data = datasets.CIFAR10(self.root, train=True, download=True)
            data_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cid = 0
        for theta in thetas:
            rotated_data = []
            labels = []
            partition = random_slicing(pre_data, shards)
            for x, y in pre_data:
                x = data_transform(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
                labels.append(y)
            for key, value in partition.items():
                data = [rotated_data[i] for i in value]
                label = [labels[i] for i in value]
                dataset = BaseDataset(data, label)
                torch.save(dataset, os.path.join(self.dir, "train", "data{}.pkl".format(cid)))
                cid += 1

        # test
        if self.dataset_name == 'mnist':
            pre_data_test = datasets.MNIST(self.root, train=False)
        elif self.dataset_name == 'cifar10':
            pre_data_test = datasets.CIFAR10(self.root, train=False)
        labels = pre_data_test.targets
        # test data is split by rotated theta group index
        for i, theta in enumerate(thetas):
            rotated_data = []
            for x, y in pre_data_test:
                x = data_transform(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            dataset = BaseDataset(rotated_data, labels)
            torch.save(dataset, os.path.join(self.dir, "test", "data{}.pkl".format(i)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader


class ShiftedPartitioner():
    def __init__(self, root, save_dir, dataset_name):
        self.root = os.path.expanduser(root)
        self.dir = save_dir
        self.dataset_name = dataset_name
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

    def pre_process(self, shards=100):
        # train
        if self.dataset_name == 'mnist':
            pre_data = datasets.MNIST(self.root, train=True)
            data_transform = transforms.ToTensor()
        elif self.dataset_name == 'cifar10':
            pre_data = datasets.CIFAR10(self.root, train=True)
            data_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        cid = 0
        for level in range(0, 10, 3):
            raw_data = []
            labels = []
            partition = random_slicing(pre_data, shards)
            for x, y in pre_data:
                x = data_transform(x)
                raw_data.append(x)
                labels.append(y)
            for key, value in partition.items():
                data = [raw_data[i] for i in value]
                label = [(labels[i] + level) % 10 for i in value]
                dataset = BaseDataset(data, label)
                torch.save(dataset, os.path.join(self.dir, "train", "data{}.pkl".format(cid)))
                cid += 1

        # test
        if self.dataset_name == 'mnist':
            pre_data_test = datasets.MNIST(self.root, train=False)
        elif self.dataset_name == 'cifar10':
            pre_data_test = datasets.CIFAR10(self.root, train=False)
        # test data is split by shifted group index
        for i, level in enumerate(range(0, 10, 3)):
            data = []
            labels = []
            for x, y in pre_data_test:
                x = data_transform(x)
                data.append(x)
                labels.append((y + level) % 10)
            dataset = BaseDataset(data, labels)
            torch.save(dataset, os.path.join(self.dir, "test", "data{}.pkl".format(i)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader


class RotatedCIFAR10Partitioner():
    def __init__(self, root, save_dir):
        self.root = os.path.expanduser(root)
        self.dir = save_dir
        # "./datasets/rotated_mnist/"
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def pre_process(self, thetas=[0, 180], shards=100):
        cifar10 = datasets.CIFAR10(self.root, train=True)

        cid = 0
        for theta in thetas:
            rotated_data = []
            partition = random_slicing(cifar10, shards)
            for x, _ in cifar10:
                x = self.transform(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            for key, value in partition.items():
                data = [rotated_data[i] for i in value]
                label = [cifar10.targets[i] for i in value]
                dataset = BaseDataset(data, label)
                torch.save(dataset, os.path.join(self.dir, "train", "data{}.pkl".format(cid)))
                cid += 1

        # test
        cifar10_test = datasets.CIFAR10(self.root, train=False)
        labels = cifar10_test.targets
        for i, theta in enumerate(thetas):
            rotated_data = []
            for x, y in cifar10_test:
                x = self.transform(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            dataset = BaseDataset(rotated_data, labels)
            torch.save(dataset, os.path.join(self.dir, "test", "data{}.pkl".format(i)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader