# -*- coding: utf-8 -*-
# @Time    : 9/27/21 12:14 AM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : original_code.py
# @Software: PyCharm
import os
import numpy as np
import copy

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_norm = 10


def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(
            torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par is None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0
    loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name),
                         batch_size=batch_size, shuffle=False)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay is not None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


def train_feddyn_mdl(model, model_func, alpha_coef,
                     avg_mdl_param,
                     local_grad_vector,
                     trn_x, trn_y,
                     learning_rate,
                     batch_size,
                     epoch,
                     print_per,
                     weight_decay,
                     dataset_name):
    n_trn = trn_x.shape[0]  # local train set的size
    trn_gen = DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                         batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')  # 损失函数

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                weight_decay=alpha_coef + weight_decay)
    model.train()
    model = model.to(device)

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(
                local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay is not None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" % (e + 1, epoch_loss))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_FedDyn(data_obj, act_prob, learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef, save_period, lr_decay_per_round,
                 rand_seed=0):
    method_name = 'FedDyn'

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x  # 每个client对应的sample
    clnt_y = data_obj.clnt_y  # 每个client对应的sample

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt

    if not os.path.exists('Output/%s/%s' % (data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' % (data_obj.name, method_name))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))  # Avg active clients
    fed_mdls_all = list(range(n_save_instances))  # Avg all clients
    fed_mdls_cld = list(range(n_save_instances))  # Cloud models

    trn_perf_sel = np.zeros((com_amount, 2))
    trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2))
    tst_perf_all = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    local_param_list = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    cld_model = model_func().to(device)
    cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    cld_mdl_param = get_mdl_params([cld_model], n_par)[0]

    if os.path.exists(
            'Output/%s/%s/%d_com_tst_perf_all.npy' % (data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load(
                'Output/%s/%s/%d_com_sel.pt' % (data_obj.name, method_name, (j + 1) * save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model

            fed_model = model_func()
            fed_model.load_state_dict(torch.load(
                'Output/%s/%s/%d_com_all.pt' % (data_obj.name, method_name, (j + 1) * save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model

            fed_model = model_func()
            fed_model.load_state_dict(torch.load(
                'Output/%s/%s/%d_com_cld.pt' % (data_obj.name, method_name, (j + 1) * save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_cld[j] = fed_model

        trn_perf_sel = np.load(
            'Output/%s/%s/%d_com_trn_perf_sel.npy' % (data_obj.name, method_name, com_amount))
        trn_perf_all = np.load(
            'Output/%s/%s/%d_com_trn_perf_all.npy' % (data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load(
            'Output/%s/%s/%d_com_tst_perf_sel.npy' % (data_obj.name, method_name, com_amount))
        tst_perf_all = np.load(
            'Output/%s/%s/%d_com_tst_perf_all.npy' % (data_obj.name, method_name, com_amount))

        clnt_params_list = np.load(
            'Output/%s/%s/%d_clnt_params_list.npy' % (data_obj.name, method_name, com_amount))
        local_param_list = np.load(
            'Output/%s/%s/%d_local_param_list.npy' % (data_obj.name, method_name, com_amount))

    else:
        for i in range(com_amount):
            # 随机生成当前轮选中的clients以及未选中的clients
            inc_seed = 0
            while True:
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True  # 选中的client从global model恢复参数

                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[
                    clnt]  # adaptive alpha coef, 根据当前client的weight调整alpha
                local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32,
                                                     device=device)  # 当前client的完整model参数
                # 用FedDyn算法对当前client的model参数进行更新
                clnt_models[clnt] = train_feddyn_mdl(model,
                                                     model_func,
                                                     alpha_coef_adpt,
                                                     cld_mdl_param_tensor,
                                                     local_param_list_curr,
                                                     trn_x, trn_y,
                                                     learning_rate * (lr_decay_per_round ** i),
                                                     batch_size,
                                                     epoch,
                                                     print_per,
                                                     weight_decay,
                                                     data_obj.dataset)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                # No need to scale up hist terms.
                # They are -\nabla/alpha and alpha is already scaled.
                local_param_list[
                    clnt] += curr_model_par - cld_mdl_param  # 在当前client上加上更新后的local model的增量
                clnt_params_list[clnt] = curr_model_par  # 更新client model参数列表中当前client的model参数

            avg_mdl_param = np.mean(clnt_params_list[selected_clnts],
                                    axis=0)  # avg_mdl参数为当前round选中client的model参数的avg
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list,
                                                    axis=0)  # cloud model参数为avg_mdl加上所有local更新的avg。 TODO: 好奇怪？

            avg_model = set_client_from_params(model_func(), avg_mdl_param)
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list,
                                                                     axis=0))  # 所有client model参数的avg
            cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            ### 评估avg model在test set上的性能
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model,
                                             data_obj.dataset)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" % (
                i + 1, acc_tst, loss_tst))
            ### 评估avg model在完整train set上的性能（把所有client上的train数据放到一起即完整train set）
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" % (
                i + 1, acc_tst, loss_tst))
            ### 评估all model在test set上的性能
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model,
                                             data_obj.dataset)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" % (
                i + 1, acc_tst, loss_tst))
            ### 评估all model在完整train set上的性能（把所有client上的train数据放到一起即完整train set）
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" % (
                i + 1, acc_tst, loss_tst))

            if ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(),
                           'Output/%s/%s/%d_com_sel.pt' % (data_obj.name, method_name, (i + 1)))
                torch.save(all_model.state_dict(),
                           'Output/%s/%s/%d_com_all.pt' % (data_obj.name, method_name, (i + 1)))
                torch.save(cld_model.state_dict(),
                           'Output/%s/%s/%d_com_cld.pt' % (data_obj.name, method_name, (i + 1)))

                np.save(
                    'Output/%s/%s/%d_local_param_list.npy' % (data_obj.name, method_name, (i + 1)),
                    local_param_list)
                np.save(
                    'Output/%s/%s/%d_clnt_params_list.npy' % (data_obj.name, method_name, (i + 1)),
                    clnt_params_list)

                np.save(
                    'Output/%s/%s/%d_com_trn_perf_sel.npy' % (data_obj.name, method_name, (i + 1)),
                    trn_perf_sel[:i + 1])
                np.save(
                    'Output/%s/%s/%d_com_tst_perf_sel.npy' % (data_obj.name, method_name, (i + 1)),
                    tst_perf_sel[:i + 1])

                np.save(
                    'Output/%s/%s/%d_com_trn_perf_all.npy' % (data_obj.name, method_name, (i + 1)),
                    trn_perf_all[:i + 1])
                np.save(
                    'Output/%s/%s/%d_com_tst_perf_all.npy' % (data_obj.name, method_name, (i + 1)),
                    tst_perf_all[:i + 1])

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' % (
                        data_obj.name, method_name, i + 1 - save_period))
                    os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' % (
                        data_obj.name, method_name, i + 1 - save_period))
                    os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' % (
                        data_obj.name, method_name, i + 1 - save_period))
                    os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' % (
                        data_obj.name, method_name, i + 1 - save_period))

                    os.remove('Output/%s/%s/%d_local_param_list.npy' % (
                        data_obj.name, method_name, i + 1 - save_period))
                    os.remove('Output/%s/%s/%d_clnt_params_list.npy' % (
                        data_obj.name, method_name, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model
                fed_mdls_all[i // save_period] = all_model
                fed_mdls_cld[i // save_period] = cld_model

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all, fed_mdls_cld
