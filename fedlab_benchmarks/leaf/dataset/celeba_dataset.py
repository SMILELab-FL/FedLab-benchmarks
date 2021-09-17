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

import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, client_id: int, client_str: str, data: list, targets: list, image_root: str, transform=None):
        """get `Dataset` for CelebA dataset

         Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): input image name list data
            targets (list):  output label list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.image_root = Path(__file__).parent.resolve() / image_root
        self.transform = transform
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _process_data_target(self):
        """process client's data and target
        """
        data = []
        targets = []
        for idx in range(len(self.data)):
            image_path = self.image_root / self.data[idx]
            image = Image.open(image_path).convert('RGB')
            data.append(image)
            targets.append(torch.tensor(self.targets[idx], dtype=torch.long))
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        target = self.targets[index]
        return data, target
