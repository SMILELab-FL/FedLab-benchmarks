# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List
from torchvision import transforms
from torch.utils.data.dataset import ConcatDataset

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from leaf.dataset.femnist_dataset import FemnistDataset
from leaf.dataset.shakespeare_dataset import ShakespeareDataset
from leaf.dataset.celeba_dataset import CelebADataset
from leaf.dataset.sent140_dataset import Sent140Dataset

logging.getLogger().setLevel(logging.INFO)


class PickleDataset:
    """Splits LEAF generated datasets and creates individual client partitions."""

    def __init__(self, dataset_name: str, data_root: str = None, pickle_root: str = None):
        """
        Args:
            dataset_name (str): name for dataset of PickleDataset Object
            data_root (str): path for data saving root.
                             Default to None and will be modified to the datasets folder in FedLab: "fedlab-benchmarks/datasets"
            pickle_root (str): path for pickle dataset file saving root.
                             Default to None and will be modified to Path(__file__).parent / "pickle_datasets"
        """
        self.dataset_name = dataset_name
        self.data_root = Path(data_root) if data_root is not None else BASE_DIR / "datasets"
        self.pickle_root = Path(pickle_root) if pickle_root is not None else Path(__file__).parent / "pickle_datasets"

    def create_pickle_dataset(self):
        # for train file data
        train_path = self.data_root / self.dataset_name / "data/train"
        original_train_datasets = sorted(list(train_path.glob("**/*.json")))
        self._read_process_json_data(dataset_type="train", paths_to_json=original_train_datasets)

        # for test file data
        test_path = self.data_root / self.dataset_name / "data/test"
        original_test_datasets = sorted(list(test_path.glob("**/*.json")))
        self._read_process_json_data(dataset_type="test", paths_to_json=original_test_datasets)

    def get_dataset_pickle(self, dataset_type: str, client_id: int = None):
        """load pickle dataset file for `dataset_name` `dataset_type` data based on client with client_id

        Args:
            dataset_type (str): Dataset type {train, test}
            client_id (int): client id. Defaults to None, which means get all_dataset pickle
        Raises:
            FileNotFoundError: No such file or directory {pickle_root}/{dataset_name}/{dataset_type}/{dataset_type}_{client_id}.pickle
        Returns:
            if there is no pickle file for `dataset`, throw FileNotFoundError, else return responding dataset
        """
        # check whether to get all datasets
        if client_id is None:
            pickle_files_path = self.pickle_root / self.dataset_name / dataset_type
            dataset_list = []
            for file in list(pickle_files_path.glob("**/*.pkl")):
                dataset_list.append(pickle.load(open(file, 'rb')))
            dataset = ConcatDataset(dataset_list)
        else:
            pickle_file = self.pickle_root / self.dataset_name / dataset_type / f"{dataset_type}_{client_id}.pkl"
            dataset = pickle.load(open(pickle_file, 'rb'))
        return dataset

    def _read_process_json_data(self, dataset_type: str, paths_to_json: List[Path]):
        """read and process LEAF generated datasets to responding Dataset
        Args:
            dataset_type (str): Dataset type {train, test}
            paths_to_json (PathLike): Path to LEAF JSON files containing dataset.
        """
        user_count = 0
        # Check whether leaf data has been downloaded
        if len(paths_to_json) == 0:
            logging.error(f"""
                            No leaf json file for {self.dataset_name} {dataset_type} data!
                            Please run leaf in `{BASE_DIR / 'datasets'}` to download origin data firstly! 
                            """)
            return

        logging.info(f"processing {self.dataset_name} {dataset_type} data to dataset in pickle file")

        for path_to_json in paths_to_json:
            with open(path_to_json, "r") as json_file:
                json_file = json.load(json_file)
                users_list = sorted(json_file["users"])
                num_users = len(users_list)
                for user_idx, user_str in enumerate(users_list):
                    self._process_user(json_file, user_count + user_idx, user_str, dataset_type)
            user_count += num_users
        logging.info(f"""
                    Complete processing {self.dataset_name} {dataset_type} data to dataset in pickle file! 
                    Located in {(self.pickle_root / self.dataset_name / dataset_type).resolve()}. 
                    All users number is {user_count}.
                    """)

    def _process_user(self, json_file: Dict[str, Any], user_idx: str, user_str: str, dataset_type: str):
        """Creates and saves partition for user
        Args:
            json_file (Dict[str, Any]): JSON file containing user data
            user_idx (str): User ID (counter) in string format
            user_str (str): Original User ID
            dataset_type (str): Dataset type {train, test}
        """
        data = json_file["user_data"][user_str]["x"]
        label = json_file["user_data"][user_str]["y"]
        if self.dataset_name == "femnist":
            dataset = FemnistDataset(client_id=user_idx,
                                     client_str=user_str,
                                     data=data,
                                     targets=label)
        elif self.dataset_name == "shakespeare":
            dataset = ShakespeareDataset(client_id=user_idx,
                                         client_str=user_str,
                                         data=data,
                                         targets=label)
        elif self.dataset_name == "celeba":
            image_size = 64
            image_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
            dataset = CelebADataset(client_id=user_idx,
                                    client_str=user_str,
                                    data=data,
                                    targets=label,
                                    image_root="../datasets/celeba/data/raw/img_align_celeba",
                                    transform=image_transform)
        elif self.dataset_name == "sent140":
            dataset = Sent140Dataset(client_id=user_idx,
                                     client_str=user_str,
                                     data=data,
                                     targets=label)

        else:
            raise ValueError("Invalid dataset:", self.dataset_name)

        # save_dataset_pickle
        save_dir = self.pickle_root / self.dataset_name / dataset_type
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f"{dataset_type}_{str(user_idx)}.pkl", "wb") as save_file:
            pickle.dump(dataset, save_file)

    def get_data_json(self, dataset_type: str):
        """ Read .json file from ``data_dir``
        This is modified by [LEAF/models/utils/model_utils.py]
        https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py

        Args:
            dataset_type (str): Dataset type {train, test}
        Returns:
            clients name dict mapping keys to id, groups list for each clients, a dict data mapping keys to client
        """
        groups = []
        client_name2data = dict()

        data_dir = self.data_root / self.dataset_name / "data" / dataset_type
        files = list(data_dir.glob("**/*.json"))
        for f in files:
            with open(f, 'r') as inf:
                cdata = json.load(inf)
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            client_name2data.update(cdata['user_data'])

        # generate clients_id_str - client_id_index map
        clients_name = list(sorted(client_name2data.keys()))
        clients_id = list(range(len(clients_name)))
        client_id2name = dict(zip(clients_id, clients_name))

        return client_id2name, groups, client_name2data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample data to build nlp vocab')
    parser.add_argument("--dataset", type=str, default='sent140')
    parser.add_argument("--data_root", type=str, default="../datasets")
    parser.add_argument("--pickle_root", type=str, default='./pickle_datasets')
    args = parser.parse_args()

    pdataset = PickleDataset(dataset_name=args.dataset, data_root=args.data_root, pickle_root=args.pickle_root)
    pdataset.create_pickle_dataset()
