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
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, client_id: int, client_str: str, input: list, output: list, image_root: str, transform=None):
        """get `Dataset` for femnist dataset

         Args:
            client_id (int): client id
            client_str (str): client name string
            input (list): input image name list data
            output (list):  output label list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.image_root = Path(__file__).parent.resolve() / image_root
        self.transform = transform
        self.data, self.targets = self.get_client_data_target(input, output)

    def get_client_data_target(self, image_names, output):
        """process client data and target for input image name and output label

        Returns: data and target for client id
        """
        data = []
        targets = []
        for idx in range(len(image_names)):
            image_path = self.image_root / image_names[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            data.append(image)
            targets.append(torch.tensor(output[idx], dtype=torch.long))
        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
