#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import re
from model.layer import BaseModel, MultiLayerPerceptron, CrossNetwork
from model.ple import PLE
from model.mmoe import MMoE
from model.pepnet import PEPNet
from model.star import STAR
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import pandas as pd
import os


class CDC(BaseModel):
    def __init__(self, feature_dims, embed_dim, n_tower, n_domain, base_model,
                 expert_dims, tower_dims, domain_idx, domain_cnt_weight=None, n_causal_mask=50, use_metric='loss',
                 device='cpu', dropout=0.2, config=None, savefig_folder='',
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5):
        super(BaseModel, self).__init__()
        self.model_name = 'cdc'
        self.base_model = base_model
        if base_model == 'mmoe':
            self.base_model_instance = MMoE(feature_dims, embed_dim, n_tower, config.mmoe_n_expert,
                                            expert_dims, tower_dims, dropout, config,
                                            l2_reg_embedding, l2_reg_linear, l2_reg_dnn, l2_reg_cross,
                                            model_name=self.model_name)
        elif base_model == 'ple':
            self.base_model_instance = PLE(feature_dims, embed_dim, n_tower,
                                           config.ple_n_expert_specific, config.ple_n_expert_shared,
                                           expert_dims, tower_dims, dropout, config,
                                           l2_reg_embedding, l2_reg_linear, l2_reg_dnn, l2_reg_cross,
                                           model_name=self.model_name)
        elif base_model == 'pepnet':
            self.base_model_instance = PEPNet(feature_dims, embed_dim, n_tower, tower_dims, config.gate_hidden_dim,
                                              domain_idx, True, dropout, config,
                                              l2_reg_embedding, l2_reg_linear, l2_reg_dnn, l2_reg_cross)
        elif base_model == 'epnet':
            self.base_model_instance = PEPNet(feature_dims, embed_dim, n_tower, tower_dims, config.gate_hidden_dim,
                                              domain_idx, False, dropout, config,
                                              l2_reg_embedding, l2_reg_linear, l2_reg_dnn, l2_reg_cross)
        elif base_model == 'star':
            self.base_model_instance = STAR(feature_dims, embed_dim,
                                            n_tower, tower_dims, domain_idx, dropout, config,
                                            l2_reg_embedding, l2_reg_linear, l2_reg_dnn, l2_reg_cross, device)

        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)
        self.device = device
        self.config = config
        self.savefig_path = os.path.join('result', config.dataset_name, savefig_folder)
        if not os.path.exists(self.savefig_path):
            os.makedirs(self.savefig_path)

        self.n_cluster = n_tower
        self.n_causal_mask = n_causal_mask
        self.n_domain = n_domain
        self.domain_idx = domain_idx
        self.domain_cnt_weight = torch.tensor(domain_cnt_weight, dtype=torch.float32, device=device)

        self.domain2group = torch.zeros(n_domain, dtype=torch.int64, device=device)
        self.domain2group_list = [0] * n_domain
        self.s_group2domain_list = [list(range(n_domain))]
        self.t_group2domain_list = [list(range(n_domain))]
        self.initial_s_group2domain_list = None
        self.call_update_group = 0
        self.p_weight = config.p_weight
        self.p_weight_method = config.p_weight_method

        self.matrix_A = torch.zeros((n_domain+1, n_domain), dtype=torch.float32, device=device)  # n_domain + 1: only warm up
        self.matrix_B = torch.zeros((n_domain+self.n_cluster, n_domain), dtype=torch.float32, device=device)  # n_domain + n_cluster: all domains in that cluster
        self.matrix_mask = torch.zeros((n_causal_mask, n_domain), dtype=torch.float32, device=device)
        self.matrix_causal = torch.zeros((n_causal_mask, n_domain), dtype=torch.float32, device=device)

        self.old_matrix_A, self.old_matrix_B, self.old_matrix_mask = None, None, None
        self.old_matrix_weight = config.old_matrix_weight

        self.use_metric = use_metric
        if (self.use_metric == 'loss') ^ (config.affinity_func == 'divide'):
            self.default_metric_value = 1e6
            self.is_max_metric_value_better = False
        else:
            self.default_metric_value = -1e6
            self.is_max_metric_value_better = True

    def forward(self, x, mode='split', domain_i=None):
        """
        mode: warmup, split
        """
        if mode == 'warmup':
            # warm-up阶段多个tower取第一个
            y_cat = self.base_model_instance.forward(x)
            return torch.mean(y_cat, dim=1)
        if mode == 'split':
            if domain_i is None:
                y_cat = self.base_model_instance.forward(x)
                groups = self.domain2group[x[:, self.domain_idx]]
                return y_cat.gather(1, groups.unsqueeze(1))
            else:
                group = self.domain2group_list[domain_i]
                y_cat = self.base_model_instance.forward(x)
                return y_cat[:, group]

    def get_matrix_metric(self, preds, targets):
        if self.use_metric == 'loss':
            res = F.binary_cross_entropy(preds, targets)
            return res.detach()
        elif self.use_metric == 'auc':
            res = roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy())
            return res

    def update_group(self, mode='iterative'):
        self.call_update_group += 1
        self.update_p_weight()
        print(f'\n======= "update_group" call_update_group-{self.call_update_group}, '
              f'mode-{mode}, affinity_func={self.config.affinity_func} =======')
        if self.old_matrix_weight > 0 and self.old_matrix_A is not None:
            print(f'old_matrix_weight: {self.old_matrix_weight}')
            self.matrix_A = self.old_matrix_A * self.old_matrix_weight + self.matrix_A * (1 - self.old_matrix_weight)
            self.matrix_B = self.old_matrix_B * self.old_matrix_weight + self.matrix_B * (1 - self.old_matrix_weight)
            # self.matrix_mask = self.old_matrix_mask * self.old_matrix_weight + self.matrix_mask * (1 - self.old_matrix_weight)

        self.old_matrix_A = copy.deepcopy(self.matrix_A)
        self.old_matrix_B = copy.deepcopy(self.matrix_B)
        self.old_matrix_mask = copy.deepcopy(self.matrix_mask)

        if self.config.affinity_func == 'minus':  # less is better
            self.matrix_A[:-1] -= self.matrix_A[-1]
            # 计算每个domain行的cluster的索引，并更新matrix_B
            self.matrix_B[:self.n_domain] = self.matrix_B[self.domain2group + self.n_domain] - self.matrix_B[:self.n_domain]
            self.matrix_mask = self.matrix_mask - self.matrix_A[-1]  # 减去纯预热的
        elif self.config.affinity_func == 'divide':  # large is better
            self.matrix_A[:-1] = 1 - self.matrix_A[:-1]/self.matrix_A[-1]
            self.matrix_B[:self.n_domain] = 1 - self.matrix_B[self.domain2group + self.n_domain]/self.matrix_B[:self.n_domain]
            self.matrix_mask = 1 - self.matrix_mask / self.matrix_A[-1]
        else:
            raise ValueError('Unknown affinity_func: ' + self.config.affinity_func)

        self.matrix_causal = self.calc_causal_matrix(self.matrix_mask.T)
        self.matrix_causal = torch.tensor(np.arccos(self.matrix_causal), dtype=torch.float32, device=self.device)

        self.save_draw_matrix(self.matrix_A, 'matrix_A', is_illustration=True)
        self.save_draw_matrix(self.matrix_B, 'matrix_B', is_illustration=True)
        self.save_draw_matrix(self.matrix_mask, 'matrix_mask', is_illustration=True)
        self.save_draw_matrix(self.matrix_causal, 'causal_matrix', is_illustration=True)

        if max(self.domain2group_list) == 0:
            # 初始化时，根据因果距离直接分组
            t_group0 = self.kmeans_group(self.matrix_causal.cpu().numpy(), self.n_cluster)
            self.domain2group_list = t_group0
            self.domain2group = torch.tensor(t_group0, dtype=torch.int64, device=self.device)
            t_group2domain_list = [[] for _ in range(self.n_cluster)]
            for i, group in enumerate(t_group0):
                t_group2domain_list[group].append(i)
            print(f't_group2domain_list0: {t_group2domain_list}')
            self.t_group2domain_list = t_group2domain_list
            self.s_group2domain_list = []
            for c in range(self.n_cluster):
                self.s_group2domain_list.append(self.get_source_domain(t_group2domain_list[c], group_idx=c))
            self.initial_s_group2domain_list = copy.deepcopy(self.s_group2domain_list)
        else:
            t_group2domain_list = self.t_group2domain_list
            print(f't_group2domain_list0: {t_group2domain_list}')
            domain_queue = list(range(self.n_domain))
            t_group, s_group = [[] for _ in range(self.n_cluster)], [[] for _ in range(self.n_cluster)]
            domain2group_metric = torch.empty(self.n_domain, self.n_cluster)
            # 计算group的中心domain
            centers = [self.get_center_domain_in_group(t_group2domain_list[c])[0] for c in range(self.n_cluster)]
            for c in range(self.n_cluster):
                t_group[c].append(centers[c])
                domain_queue.remove(centers[c])
                domain2group_metric[centers[c], :] = self.default_metric_value

            if mode == 'iterative':
                is_domain_queue_update = True
                while domain_queue and is_domain_queue_update:
                    is_domain_queue_update = False
                    for c in range(self.n_cluster):
                        s_group[c] = self.get_source_domain(t_group[c], group_idx=c)
                    for d in domain_queue:
                        for c in range(self.n_cluster):
                            domain2group_metric[d, c] = self.calc_metric_in_source_group(d, s_group[c])
                    if self.is_max_metric_value_better:
                        best_domain = torch.argmax(domain2group_metric, dim=0)  # shape: n_cluster
                    else:
                        best_domain = torch.argmin(domain2group_metric, dim=0)
                    # print(f'best_domain: {best_domain}')
                    for c in range(self.n_cluster):
                        if self.is_max_metric_value_better:
                            flag = (torch.argmax(domain2group_metric[best_domain[c], :]) == c)
                        else:
                            flag = (torch.argmin(domain2group_metric[best_domain[c], :]) == c)
                        if flag:
                            is_domain_queue_update = True
                            best_domain_int = best_domain[c].item()
                            t_group[c].append(best_domain_int)
                            domain_queue.remove(best_domain_int)
                            domain2group_metric[best_domain[c], :] = self.default_metric_value
                if domain_queue:
                    print(f'domain_queue: {domain_queue}')
                    print('domain2group_metric:', domain2group_metric[domain_queue, :])
                    raise ValueError('target domain_queue is not empty')
            elif mode == 'greedy':
                for c in range(self.n_cluster):
                    s_group[c] = self.get_source_domain(t_group[c], group_idx=c)
                for d in domain_queue:
                    for c in range(self.n_cluster):
                        domain2group_metric[d, c] = self.calc_metric_in_source_group(d, s_group[c])
                if self.is_max_metric_value_better:
                    for d in domain_queue:
                        best_group = torch.argmax(domain2group_metric[d, :])
                        t_group[best_group].append(d)
                else:
                    for d in domain_queue:
                        best_group = torch.argmin(domain2group_metric[d, :])
                        t_group[best_group].append(d)

            self.t_group2domain_list = t_group
            print(f't_group2domain_list updated: {t_group}')
            domain2group_array = np.array([0] * self.n_domain)
            for c in range(self.n_cluster):
                self.s_group2domain_list[c] = self.get_source_domain(t_group[c], group_idx=c)
                domain2group_array[t_group[c]] = c
            domain2group_array = domain2group_array.astype(int)
            self.domain2group = torch.tensor(domain2group_array, dtype=torch.int64, device=self.device)
            self.domain2group_list = domain2group_array.tolist()
        print(f'domain2group_list: {self.domain2group_list}')
        print(f's_group2domain_list: {self.s_group2domain_list}')
        return self.domain2group_list

    def get_source_domain(self, t_group, group_idx):
        # print(f'======= "get_source_domain" call_update_group-{self.call_update_group}, group_idx-{group_idx}, t_group: {t_group} =======')
        # init
        s_group = self.get_center_domain_in_group(t_group, center_num=2)
        has_useful_domain = True

        while has_useful_domain and len(s_group) < self.n_domain:
            # 场景d_i对测试场景d_t的增益
            lambda_t_k = []
            for d_i in range(self.n_domain):
                if d_i in s_group:
                    lambda_i_t_k = torch.zeros(len(t_group), dtype=torch.float32, device=self.device)
                else:
                    lambda_i_t_k = self.calc_domain_lambda_in_group(group=s_group+[d_i], domain=t_group)  # calc lambda of t_group domains
                lambda_t_k.append(lambda_i_t_k)
            lambda_t_k = torch.stack(lambda_t_k, dim=0)
            assert lambda_t_k.shape == (self.n_domain, len(t_group))

            domain_weights_adjusted = self.domain_cnt_weight[t_group]
            sum_weights = domain_weights_adjusted.sum()
            if sum_weights != 0:
                domain_weights_adjusted = domain_weights_adjusted/sum_weights

            A_selected = self.matrix_A[:self.n_domain, t_group]
            B_selected = self.matrix_B[:self.n_domain, t_group]
            J = (((1 - lambda_t_k) * A_selected + lambda_t_k * B_selected)*domain_weights_adjusted).sum(dim=1)  # shape: n_domain
            assert J.shape[0] == self.n_domain

            if self.initial_s_group2domain_list is None:
                result = J
                # print('result:', result)
            else:
                # 计算d_i在最终的S的可能性，越大越好
                P = (1 - 2*self.calc_domain_lambda_in_group(
                    group=self.initial_s_group2domain_list[group_idx])) * torch.pow(self.domain_cnt_weight, 0.5)
                P_weight = self.p_weight
                if self.is_max_metric_value_better:
                    result = J + P_weight * P
                else:
                    result = J - P_weight * P
                # print('J:', J)
                # print('P:', P)
                # print('P_weight:', P_weight)
                # print('result:', result)
            result[s_group] = self.default_metric_value
            if self.is_max_metric_value_better:
                best_value, best_domain = torch.max(result, 0)
                has_useful_domain = (best_value > 0)
            else:
                best_value, best_domain = torch.min(result, 0)
                has_useful_domain = (best_value < 0)
            if has_useful_domain:
                # print(f'best_domain: {best_domain.item()}')
                s_group.append(best_domain.item())
        # print(f't_group: {t_group}, get_source_domain: {s_group}')
        # print(f'======= end "get_source_domain" call_update_group-{self.call_update_group}, group_idx-{group_idx} =======')
        return s_group

    def update_p_weight(self):
        if self.p_weight > 1e-10:
            if self.p_weight_method == 'linear_decay':
                self.p_weight = self.config.p_weight / self.call_update_group
            elif self.p_weight_method == 'quadratic_decay':
                self.p_weight = self.config.p_weight / (self.call_update_group ** 2)
            elif self.p_weight_method == 'exponential_decay':
                self.p_weight = self.p_weight * self.config.p_weight_exp_decay
        print('call_update_group:', self.call_update_group, 'p_weight:', self.p_weight)

    def calc_metric_in_source_group(self, target_domain, s_group):
        lambda_domain = self.calc_domain_lambda_in_group(group=s_group, domain=[target_domain])
        domain_metric = torch.sum((1-lambda_domain) * self.matrix_A[s_group, target_domain] +
                                  lambda_domain * self.matrix_B[s_group, target_domain])
        return domain_metric

    def get_center_domain_in_group(self, group, center_num=1):
        center_num = min(center_num, len(group))
        domain_distance = self.calc_domain_lambda_in_group(group=group, domain=group)
        best_values, best_domains = torch.topk(domain_distance, k=center_num, largest=False)
        # print(f'"get_center_domain_in_group" group: {group}, center_num: {center_num}, best_domains: {best_domains}')
        return [group[i] for i in best_domains]

    def calc_domain_lambda_in_group(self, group, domain=None, mode='avg_dis'):
        if mode == 'avg_dis':
            group_dis = self.matrix_causal[np.ix_(group, group)]
            group_total_dis = torch.sum(group_dis)

            if domain is None:
                domain = list(range(self.n_domain))

            domain_related_dis = torch.sum(self.matrix_causal[np.ix_(group, domain)], dim=0)
            non_related_dis = group_total_dis - domain_related_dis
            domain_values = (len(group) - 1) * domain_related_dis / non_related_dis * 0.5

            domain_similar_values = torch.clamp(domain_values, min=0, max=1)
            assert domain_similar_values.shape[0] == len(domain)

            # if domain is None:
            #     domain_similar = torch.sum(self.matrix_causal[group], dim=0)  # shape: n_domain
            # else:
            #     domain_similar = torch.sum(self.matrix_causal[np.ix_(group, domain)], dim=0)  # shape: len(domain)
            # return torch.clamp(domain_similar / group_similar, min=0, max=1)
            return domain_similar_values

    def save_model_state(self):
        params_to_save = ['base_model_instance']
        regex_pattern = '^(' + '|'.join(params_to_save) + ')'
        pattern = re.compile(regex_pattern)

        full_state_dict = self.state_dict()
        # 使用正则表达式进行匹配
        selected_state_dict = {k: v for k, v in full_state_dict.items() if pattern.match(k)}
        self.model_state = copy.deepcopy(selected_state_dict)

    def load_model_state(self):
        self.load_state_dict(self.model_state, strict=False)

    def get_regularization_loss(self, device):
        return self.base_model_instance.get_regularization_loss(device)

    @staticmethod
    def kmeans_group(matrix_causal, n_cluster):
        kmeans = KMeans(n_clusters=n_cluster).fit(matrix_causal)
        return kmeans.labels_

    @staticmethod
    def calc_causal_matrix(X, alpha=None):
        """
        References: A Distance Covariance-based Kernel for Nonlinear Causal Clustering in Heterogeneous Populations
        https://causal.dev/code/dep_con_kernel.py
        """
        if not isinstance(X, np.ndarray):
            X = X.cpu().numpy()
        num_samps, num_feats = X.shape
        thresh = np.eye(num_feats)
        if alpha is not None:
            thresh[thresh == 0] = (
                    chi2(1).ppf(1 - alpha) / num_samps
            )  # critical value corresponding to alpha
            thresh[thresh == 1] = 0
        Z = np.zeros((num_feats, num_samps, num_samps))
        for j in range(num_feats):
            D = squareform(pdist(X[:, j].reshape(-1, 1), "cityblock"))
            # doubly center and standardized:
            Z[j] = ((D - D.mean(0) - D.mean(1).reshape(-1, 1)) / D.mean()) + 1

        F = Z.reshape(num_feats * num_samps, num_samps)
        left = np.tensordot(Z, thresh, axes=([0], [0]))
        left_right = np.tensordot(left, Z, axes=([2, 1], [0, 1]))
        gamma = (F.T @ F) ** 2 - 2 * (left_right) + np.linalg.norm(thresh)  # helper kernel

        diag = np.diag(gamma)
        kappa = gamma / np.sqrt(np.outer(diag, diag))  # cosine similarity
        kappa[kappa > 1] = 1  # correct numerical errors
        return kappa

    def save_draw_matrix(self, matrix, name, is_illustration=False):
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().numpy()
        df = pd.DataFrame(matrix)
        save_path = os.path.join(self.savefig_path, f'{name} step-{self.call_update_group}.xlsx')
        df.to_excel(save_path, index=False)

        if is_illustration:
            if 'A' in name or 'B' in name:
                matrix = matrix[:self.n_domain]
            n_row, n_col = matrix.shape
            if n_row <= self.n_domain:
                plt.figure(figsize=(20, 16))
            else:
                plt.figure(figsize=(20, int(self.n_causal_mask*0.7)))
            v_abs_max = max(abs(matrix.min()), abs(matrix.max()))

            c = plt.imshow(matrix, cmap='RdBu', interpolation='nearest', vmin=-v_abs_max, vmax=v_abs_max)
            plt.title(f'{name} step-{self.call_update_group}', fontsize=20)
            plt.colorbar(c)

            plt.xlabel('Domain Index', fontsize=16)
            plt.ylabel('Treatment Index', fontsize=16)
            plt.xticks(range(n_col), fontsize=12)
            plt.yticks(range(n_row), fontsize=12)

            for i in range(n_row):
                for j in range(n_col):
                    plt.text(j, i, f"{matrix[i, j]:.1e}", ha='center', va='center', color='black', fontsize=8)

            plt.savefig(os.path.join(self.savefig_path, f'{name} step-{self.call_update_group}.png'))
            plt.close()








