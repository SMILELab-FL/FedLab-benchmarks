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

"""
    get dataloader for dataset in LEAF processed
"""

import torch
from torch.utils.data import ConcatDataset
from .read_util import get_dataset_pickle
from .nlp_utils.dataset_vocab.sample_build_vocab import get_built_vocab


def get_LEAF_dataloader(dataset: str, client_id=0, batch_size=128):
    """Get dataloader with ``batch_size`` param for client with ``client_id``

    Args:
        dataset (str):  dataset name string to get dataloader
        client_id (int, optional): assigned client_id to get dataloader for this client. Defaults to 0
        batch_size (int, optional): the number of batch size for dataloader. Defaults to 128

    Returns:
        A tuple with train dataloader and test dataloader for the client with `client_id`

    Examples:
        trainloader, testloader = get_LEAF_dataloader(dataset='shakespeare', client_id=1)
    """
    # get vocab and index data
    if dataset == 'sent140':
        vocab = get_built_vocab(dataset)

    pickle_root = "./process_data/pickle_dataset"
    trainset = get_dataset_pickle(dataset_name=dataset, client_id=client_id, dataset_type='train', pickle_root=pickle_root)
    testset = get_dataset_pickle(dataset_name=dataset, client_id=client_id, dataset_type='test', pickle_root=pickle_root)

    # get vocab and index data
    if dataset == 'sent140':
        vocab = get_built_vocab(dataset)
        trainset.token2seq(vocab, maxlen=300)
        testset.token2seq(vocab, maxlen=300)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        drop_last=False)  # avoid train dataloader size 0
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=len(testset),
        drop_last=False,
        shuffle=False)
    
    return trainloader, testloader


def get_LEAF_all_test_dataloader(dataset: str):
    """Get dataloader for all clients' test pickle file

    Args:
        dataset (str): dataset name

    Returns:
        ConcatDataset for all clients' test dataset
    """
    all_testset = get_LEAF_all_test_dataloader(dataset=dataset)
    return all_testset
