# -*- coding: utf-8 -*-
# @Time    : 9/27/21 12:49 AM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : config.py
# @Software: PyCharm

cifar10_config = {
    'num_clients': 100,
    'model_name': 'Cifar10Net',  # Model type
    'com_amount': 1000,
    'save_period': 200,
    'weight_decay': 1e-3,
    'batch_size': 50,
    'act_prob': 1,
    'lr_decay_per_round': 1,
    'epoch': 5,
    'lr': 0.1,
    'print_per': 5,
    'alpha_coef': 1e-2,
    'max_norm': 10,
}

balance_iid_data_config = {
    'partition': "iid",
    'balance': True,
    'dataset': 'cifar10',
    'num-clients': 100,
}
