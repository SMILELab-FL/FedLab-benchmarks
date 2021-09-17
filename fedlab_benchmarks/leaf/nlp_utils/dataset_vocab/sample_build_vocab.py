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

import os
import math
import random
import pickle
import argparse
import sys
from pathlib import Path

from ...read_util import get_data_json
from ...nlp_utils.tokenizer import Tokenizer
from ...nlp_utils.vocab import Vocab


class DataSample:
    """ sample some train data to generate vocab in nlp question
    Args:
        dataset (str): string of dataset name
        data_root (str): string path for data saving root
        select_ratio (float): select ratio
        is_to_tokens (bool, optional): check if using tokenizer , default to True
        tokenizer (bool, optional): giving tokenizer to do tokenization
    """
    def __init__(self, dataset: str, data_root: str, select_ratio: float, is_to_tokens=True, tokenizer=None):
        self.dataset = dataset
        self.data_root = data_root
        self.select_ratio = select_ratio
        self.select_client, self.data = self.choose_client_data()
        self.data_token = []
        self.tokenizer = tokenizer if tokenizer else Tokenizer('normal')

        if is_to_tokens:
            self.data2token()

    def choose_client_data(self):
        client_id_name_dict, client_groups, client_name_data_dict = get_data_json(data_root=self.data_root,
                                                                                  dataset_name=self.dataset,
                                                                                  dataset_type="train")

        client_num = len(client_id_name_dict)
        random.seed(0)
        select_client = random.sample(range(client_num), math.floor(self.select_ratio * client_num))
        data = []

        for client_id in select_client:
            client_name = client_id_name_dict[client_id]
            # choose the first data to build vocab
            data.append(self.__process_x(client_name_data_dict[client_name]['x'][0]))

        return select_client, data

    def data2token(self):
        assert self.data is not None
        for sen in self.data:
            self.data_token.append(self.tokenizer(sen))

    def __process_x(self, raw_x):
        if self.dataset == 'sent140':
            raw_x = raw_x[4]
        return raw_x


def build_vocab(data_root: str, dataset: str, data_select_ratio: float, vocab_limit_size: int, save_root: str):
    """Build vocab for dataset with random selected client

    Args:
        data_root (str): string path for data saving root
        dataset (str): string of dataset name to build vocab
        data_select_ratio (float): random select clients ratio
        vocab_limit_size (int): limit max number of vocab size
        save_root (str): string of path to save built vocab
    Returns:
        save vocab.pck for dataset
    """
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    save_file_path = save_root / f"{dataset}_vocab.pickle"
    if save_file_path.exists():
        print('There has been a built vocab file for {} dataset in {}, '
              'please delete it before re-building'.format(dataset, Path))
        return

    data_sample = DataSample(dataset=dataset, data_root=data_root, select_ratio=data_select_ratio)
    vocab = Vocab(origin_data_tokens=data_sample.data_token, vocab_limit_size=vocab_limit_size)
    with open(save_file_path, "wb") as save_file:
        pickle.dump(vocab, save_file)
    print('sample data to build vocab for {} dataset is completed!'.format(dataset))


def get_built_vocab(dataset: str) -> Vocab:
    """load vocab file for `dataset` to get Vocab based on selected client and data in current directory

    Args:
        dataset (str): string of dataset name to get vocab
    Returns:
        if there is no built vocab file for `dataset`, return None, else return Vocab
    """
    vocab_file_path = Path(__file__).parent.resolve() / f'{dataset}_vocab.pickle'
    if not vocab_file_path.exists():
        print('There is no built vocab file for {} dataset, please run `main` or `build_vocab.sh` to build it firstly.'
              .format(dataset))
        return None
    vocab_file = open(vocab_file_path, 'rb')  # get vocab based on sample data
    vocab = pickle.load(vocab_file)
    return vocab


# Example: python sample_build_vocab.py --dataset "sent140" --data_select_ratio 0.25 --vocab_limit_size 30000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample data to build nlp vocab')
    parser.add_argument("--data_root", type=str, default='../../../datasets/data')
    parser.add_argument("--dataset", type=str, default='sent140')
    parser.add_argument("--data_select_ratio", type=float, default=0.25)
    parser.add_argument("--vocab_limit_size", type=int, default=30000)
    parser.add_argument("--save_root", type=str, default='./')
    args = parser.parse_args()

    build_vocab(data_root=Path(args.data_root), dataset=args.dataset, data_select_ratio=args.data_select_ratio,
                vocab_limit_size=args.vocab_limit_size, save_root=args.save_root)
