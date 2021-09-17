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
import torch
from torch.utils.data import Dataset


class FemnistDataset(Dataset):

    def __init__(self, client_id: int, client_str: str, data: list, targets: list):
        """get `Dataset` for femnist dataset

         Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): image data list
            targets (list): image class target list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _process_data_target(self):
        """process client's data and target

        """
        self.data = torch.tensor(self.data, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
