import torch

import os
import fnmatch
import numpy as np
import sys

sys.path.append("../../../FedLab/")

from fedlab.utils.functional import AverageMeter

from config import local_grad_vector_list_file_pattern


def load_local_grad_vector(out_dir, rank=None):
    local_grad_vector_list = []
    if rank is not None:
        # if rank is specified, read local_grad
        local_grad_vector_list = torch.load(
            os.path.join(out_dir, local_grad_vector_list_file_pattern.format(rank=rank)))
    else:
        # if rank=None, then read all files matching with patterns
        tmp_files = os.listdir(out_dir)
        local_grad_vector_list_files = sorted(fnmatch.filter(tmp_files,
                                                             local_grad_vector_list_file_pattern.replace(
                                                                 '{rank:02d}', '*')))
        for fn in local_grad_vector_list_files:
            fn = os.path.join(out_dir, fn)
            local_grad_vector_list.extend(torch.load(fn))
    assert len(local_grad_vector_list) >= 1
    return local_grad_vector_list


def evaluate(model, criterion, test_loader):
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.avg, acc_.avg
