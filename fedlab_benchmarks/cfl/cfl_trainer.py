import torch
from fedlab.utils import SerializationTool
from tqdm import tqdm
from fedlab.core.client.serial_trainer import SerialTrainer

from fedlab_benchmarks.cfl.datasets import ShiftedPartitioner, RotatedPartitioner
from fedlab_benchmarks.leaf.pickle_dataset import PickleDataset


class CFLTrainer(SerialTrainer):
    def __init__(self, model, args, cuda=True, logger=None):
        super().__init__(model, None, cuda, logger)
        self.args = args
        if self.args.dataset == "femnist":
            self.client_num = 3597  # femnist
            self.dataset = PickleDataset(dataset_name="femnist", data_root=args.root,
                                         pickle_root=args.save_dir)
            self.weights = [len(self.dataset.get_dataset_pickle("train", i)) for i in range(self.client_num)]

        elif self.args.dataset == "mnist" or self.args.dataset == "cifar10":
            self.client_num = self.args.n
            if self.args.augment == 'shifted':
                self.dataset = ShiftedPartitioner(root=args.root, save_dir=args.save_dir,
                                                  dataset_name=self.args.dataset)
            elif self.args.augment == 'rotated':
                self.dataset = RotatedPartitioner(root=args.root, save_dir=args.save_dir,
                                                  dataset_name=self.args.dataset)
            if self.args.process_data == 1:
                print("---Process data---")
                self.dataset.pre_process(shards=self.client_num // 4)
                print("---Process data end---")
            self.weights = [len(self.dataset.get_dataset(i, "train")) for i in range(self.client_num)]

    def _get_dataloader(self, client_id):
        if self.args.dataset == 'femnist':
            trainset = self.dataset.get_dataset_pickle("train", client_id)
            data_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=self.args.batch_size,
                drop_last=False)
        elif self.args.dataset == "mnist" or self.args.dataset == "cifar10":
            data_loader = self.dataset.get_data_loader(client_id, batch_size=self.args.batch_size, type="train")

        return data_loader

    def get_testloader(self, client_id):
        if self.args.dataset == "mnist" or self.args.dataset == "cifar10":
            id = int(client_id / (self.client_num // 4))
            dataset = self.dataset.get_dataset(id, type='test')
            data_loader = self.dataset.get_data_loader(id, batch_size=4096, type="test")
            return len(dataset), data_loader

    def local_process(self, id_list, model_parameters):
        updates, dWs = [], []
        for id in tqdm(id_list):
            data_loader = self._get_dataloader(id)
            dw = self._train_alone(model_parameters, data_loader, id)
            updates.append(self.model_parameters)
            dWs.append(dw)
        return updates, dWs

    def _train_alone(self, model_parameters, train_loader, id):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)
        SerializationTool.deserialize_model(self._model, model_parameters)
        self._model.cuda(self.gpu)
        self._model.train()

        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                loss = criterion(preds, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model_parameters - self.model_parameters  # dW

