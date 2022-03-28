from sys import path

path.append("../../")

from torchvision import datasets
import ssl
import os

# for solving http error raised by download dataset
root = "../../datasets/emnist"
ssl._create_default_https_context = ssl._create_unverified_context
if os.path.isdir(root) is False:
    os.mkdir(root)

datasets.EMNIST(root, split="byclass", download=True)
