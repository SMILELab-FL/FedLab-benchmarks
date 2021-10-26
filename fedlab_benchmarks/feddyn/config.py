cifar10_config = {
    'num_clients': 100,
    'model_name': 'Cifar10Net',  # Model type
    'round': 1000,
    'save_period': 200,
    'weight_decay': 1e-3,
    'batch_size': 50,
    'test_batch_size': 256,  # no this param in official code
    'lr_decay_per_round': 1,
    'epochs': 5,
    'lr': 0.1,
    'print_freq': 5,
    'alpha_coef': 1e-2,
    'max_norm': 10,
    'sample_ratio': 1,
    'partition': 'iid',
    'dataset': 'cifar10',
}

balance_iid_data_config = {
    'partition': "iid",
    'balance': True,
    'dataset': 'cifar10',
    'num-clients': 100,
}

debug_config = {
    'num_clients': 30,
    'model_name': 'Cifar10Net',  # Model type
    'round': 5,
    'save_period': 2,
    'weight_decay': 1e-3,
    'batch_size': 256,
    'test_batch_size': 256,
    'act_prob': 1,
    'lr_decay_per_round': 1,
    'epochs': 2,
    'lr': 0.1,
    'print_freq': 1,
    'alpha_coef': 1e-2,
    'max_norm': 10,
    'sample_ratio': 1,
    'partition': 'iid',
    'dataset': 'cifar10'
}

# usage: local_params_file_pattern.format(cid=cid)
local_grad_vector_file_pattern = "client_{cid:03d}_local_grad_vector.pt"  # accumulated model gradient
clnt_params_file_pattern = "client_{cid:03d}_clnt_params.pt"  # latest model param

local_grad_vector_list_file_pattern = "client_rank_{rank:02d}_local_grad_vector_list.pt"  # accumulated model gradient for clients in one client process
clnt_params_list_file_pattern = "client_rank_{rank:02d}_clnt_params_list.pt"  # latest model param for clients in one client process
