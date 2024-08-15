#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
import os
import ast
import pickle
from sklearn.metrics import roc_auc_score, log_loss
from datetime import datetime
import wandb
from model.dfm import DeepFM
from model.dcn import DCN
from model.dcnv2 import DCNv2
from model.autoint import AutoInt
from model.ple import PLE
from model.mmoe import MMoE
from model.pepnet import PEPNet
from model.star import STAR
from model.cdc import CDC
from model.adl import ADL
from model.hinet import HiNet
from model.adasparse import AdaSparse
from dataset.aliccp.preprocess_ali_ccp import reduce_mem


class Run(object):
    def __init__(self, config):
        device = 'cuda:' + str(config.gpu) if config.use_cuda and torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = config.model
        self.base_model = config.base_model
        self.epoch = config.epoch
        self.embed_dim = config.embed_dim
        # 当model包含聚类算法时，使用mix混合数据再根据n_cluster进行聚类；为其他模型时，使用给定聚类
        self.n_cluster = config.n_cluster
        self.domain2encoder_dict = config.domain2encoder_dict
        self.domain2group_list = config.domain2group_org_dict[config.dataset_name][config.group_strategy]
        self.domain2group_dict = {k: self.domain2group_list[k] for k in range(len(self.domain2encoder_dict))}
        self.n_tower = config.n_cluster if self.model in ['cdc', 'adl'] else max(self.domain2group_list) + 1
        self.n_domain = None
        self.preprocess_path = config.preprocess_path
        self.domain_filter = getattr(config, 'domain_filter', None)
        self.config = config
        self.preprocess_folder = self.preprocess_path.split('.csv')[0]
        self.dataset_name = config.dataset_name
        if self.dataset_name == 'amazon':
            self.feature_names = ['userid', 'itemid', 'weekday', 'domain', 'sales_chart', 'sales_rank', 'brand', 'price']
            self.label_name = 'label'
        elif self.dataset_name == 'aliccp':
            categorical_columns = ['userid', '121', '122', '124', '125', '126', '127', '128', '129', 'itemid', 'domain',
                                   '207', '210', '216', '508', '509', '702', '853', '109_14', '110_14', '127_14',
                                   '150_14', '301']
            numerical_columns = []  # ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
            self.feature_names = categorical_columns + numerical_columns
            self.label_name = 'click'
            if len(numerical_columns) == 0:
                print("Warning: no using numerical columns in aliccp dataset")
        self.feature_dims = None
        self.itemid_idx, self.domain_idx = None, None
        self.is_multi_tower = (self.model in ['ple', 'mmoe', 'pepnet', 'epnet', 'star', 'adl', 'adl-split', 'hinet'])
        self.is_concat_group = (self.model in ['star', 'adl', 'adl-split', 'hinet'])
        self.selected_domain, self.train_loss_domain_group = None, None
        self.train_valid, self.valid_test = None, None
        self.domain_cnt_weight = None
        self.train_domain_batch_seq, self.valid_domain_batch_seq, self.test_domain_batch_seq = [], [], []
        self.train_data_loader, self.valid_data_loader, self.test_data_loader = None, None, None
        self.train_data_generator, self.valid_data_generator, self.test_data_generator = None, None, None

        # find the latest model
        # all_models = [f for f in os.listdir(self.config.save_path) if f.startswith(f'{self.model}') and f.endswith('.pth.tar')]
        # if all_models:
        #     latest_model_inx = max([int(x.split('_')[1].split('.')[0]) for x in all_models])
        # else:
        #     latest_model_inx = 0
        self.latest_model_inx = np.random.randint(50)
        self.save_model_path = os.path.join(self.config.save_path, f'{self.model}_{self.latest_model_inx+1}.pth.tar')
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
            print(f'create save_path folder: {self.config.save_path}')
        print('save_model_path: ', self.save_model_path)

        # for early stop
        self.num_trials = config.early_stop
        self.trial_counter = 0
        self.best_loss, self.best_mean_loss = np.inf, np.inf
        self.best_auc, self.best_mean_auc = 0, 0

        wandb.log({'domain2group_list': self.domain2group_list, 'n_tower': self.n_tower})

    def read_split_data(self, path, only_id=False):
        if only_id:
            x_cols = ['userid', 'itemid', 'domain']
            self.feature_names = x_cols
            print('only id features and domain')
        else:
            x_cols = self.feature_names

        print('feature_names: ', self.feature_names)
        y_col = [self.label_name]
        return_cols = x_cols + y_col
        if self.dataset_name == 'amazon':
            # 使用timestamp作为划分数据集的依据
            split_col = 'timestamp'
        elif self.dataset_name == 'aliccp':
            # preprocess_ali_ccp.py中已经将数据集划分好了，预处理时打了train_tag标签
            split_col = 'train_tag'  # train_tag: 0-train, 1-val, 2-test
        cols = x_cols + y_col + [split_col]

        data = reduce_mem(pd.read_csv(path, usecols=cols))
        # data.sort_values(by=['timestamp', 'domain'], inplace=True)
        if self.dataset_name == 'amazon':
            self.train_valid, self.valid_test = data[split_col].quantile(0.9), data[split_col].quantile(0.95)
        else:
            self.train_valid, self.valid_test = 1, 2

        if self.domain_filter is not None:
            self.domain_filter = ast.literal_eval(self.domain_filter)
            print(f'filter domain: {self.domain_filter}')
            data = data.loc[data['domain'].isin(self.domain_filter)].copy()

        self.itemid_idx = x_cols.index('itemid')
        self.domain_idx = x_cols.index('domain')
        self.feature_dims = np.max(data[x_cols], axis=0) + 1  # length of one-hot
        self.n_domain = data['domain'].nunique()

        if self.is_multi_tower:
            print(f'group num in multi-tower: {self.n_tower}')
            print('domain2group_list: ', self.domain2group_list)

        wandb.log({'preprocess_folder': self.preprocess_folder,
                   'save_model_path': self.save_model_path,
                   'feature_names': self.feature_names})

        if self.domain_filter is None:  # (self.config.is_set_seed == 0) and (self.domain_filter is None):
            # 高效跑程序模式（无随机种子）且不需要筛domain->不做数据统计，直接读取预处理好的数据
            del data
            return return_cols, (None, None, None)

        if self.config.is_evaluate_multi_domain:
            damain_counts = data['domain'].value_counts()
            domain_positive = data.groupby(by='domain')[self.label_name].sum()
            domain_statistics = pd.concat([damain_counts, domain_positive], axis=1)
            print('counts of each domain in total data:')
            print(domain_statistics)

        train_data = data[data[split_col] < self.train_valid]
        valid_data = data[(data[split_col] >= self.train_valid) & (data[split_col] < self.valid_test)]
        test_data = data[data[split_col] >= self.valid_test]

        data_len = [train_data.shape[0], valid_data.shape[0], test_data.shape[0]]
        print('before multi-hot features flatten, including input features, timestamps and labels')
        print(f'train_data:{train_data.shape}, valid_data:{valid_data.shape}, test_data:{test_data.shape}')
        print(f'train:valid:test = '
              f'{data_len[0]/sum(data_len):.2f}:{data_len[1]/sum(data_len):.2f}:{data_len[2]/sum(data_len):.2f}')
        if self.dataset_name == 'amazon':
            print(f'train time: {datetime.fromtimestamp(train_data["timestamp"].min()).strftime("%Y-%m-%d")} '
                  f'to {datetime.fromtimestamp(train_data["timestamp"].max()).strftime("%Y-%m-%d")}')
            print(f'valid time: {datetime.fromtimestamp(valid_data["timestamp"].min()).strftime("%Y-%m-%d")} '
                  f'to {datetime.fromtimestamp(valid_data["timestamp"].max()).strftime("%Y-%m-%d")}')
            print(f'test  time: {datetime.fromtimestamp(test_data["timestamp"].min()).strftime("%Y-%m-%d")} '
                  f'to {datetime.fromtimestamp(test_data["timestamp"].max()).strftime("%Y-%m-%d")}')

        # count overlap
        print('Calculate the user overlap between train_data, valid_data and test_data')
        train_user_ids = set(train_data['userid'].unique())
        valid_user_ids = set(valid_data['userid'].unique())
        test_user_ids = set(test_data['userid'].unique())
        train_valid_user_inter = len(train_user_ids.intersection(valid_user_ids))
        train_test_user_inter = len(train_user_ids.intersection(test_user_ids))
        print(f'{train_valid_user_inter}/{len(valid_user_ids)} ({(train_valid_user_inter/len(valid_user_ids)):.2f}) '
              f'users in valid_data exists in train_data')
        print(f'{train_test_user_inter}/{len(test_user_ids)} ({(train_test_user_inter/len(test_user_ids)):.2f}) '
              f'users in test_data exists in train_data')

        print('Calculate the item overlap between train_data, valid_data and test_data')
        train_item_ids = set(train_data['itemid'].unique())
        valid_item_ids = set(valid_data['itemid'].unique())
        test_item_ids = set(test_data['itemid'].unique())
        train_valid_item_inter = len(train_item_ids.intersection(valid_item_ids))
        train_test_item_inter = len(train_item_ids.intersection(test_item_ids))
        print(f'{train_valid_item_inter}/{len(valid_item_ids)} ({(train_valid_item_inter/len(valid_item_ids)):.2f}) '
              f'items in valid_data exists in train_data')
        print(f'{train_test_item_inter}/{len(test_item_ids)} ({(train_test_item_inter/len(test_item_ids)):.2f}) '
              f'items in test_data exists in train_data')

        return return_cols, (train_data[return_cols],
                             valid_data[return_cols],
                             test_data[return_cols])

    def save_tensor_from_data(self, data, cols, mode):
        x_cols = [col for col in cols if col != self.label_name]
        y_col = [self.label_name]
        print('save_tensor_from_data: x_cols:', x_cols)
        X = torch.tensor(data[x_cols].values, dtype=torch.int)
        y = torch.tensor(data[y_col].values, dtype=torch.short)
        if not os.path.exists(self.preprocess_folder):
            os.makedirs(self.preprocess_folder)
        torch.save(X, os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth'))
        torch.save(y, os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth'))

        return X, y

    def convert2data_loader(self, data, cols, mode):
        cols = cols if data is None else data.columns
        x_cols = [col for col in cols if col != self.label_name]
        y_col = [self.label_name]

        if os.path.exists(os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth')):
            X = torch.load(os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth')).to(torch.int)
            y = torch.load(os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth')).to(torch.short)
        elif os.path.exists(os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth')):
            X = torch.load(os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth')).to(torch.int)
            y = torch.tensor(data[y_col].values, dtype=torch.short)
            torch.save(y, os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth'))
        else:
            X, y = self.save_tensor_from_data(data, cols, mode)

        assert X.shape[1] == len(x_cols), f'X.shape[1] != len(x_cols): {X.shape[1]} != {len(x_cols)}'

        if self.domain_filter is not None:
            mask = torch.isin(X[:, self.domain_idx], torch.tensor(self.domain_filter))
            X = X[mask]
            y = y[mask]
        if self.is_multi_tower:
            group = pd.Series(X[:, self.domain_idx]).map(self.domain2group_dict)
            group = torch.tensor(group.values, dtype=torch.int64).view(-1, 1).to(self.device)

        if self.config.is_evaluate_multi_domain and mode == 'train':
            domain_val = X[:, self.domain_idx]
            domain_cnt = domain_val.bincount()
            self.domain_cnt_weight = np.array([domain_cnt[i]/X.shape[0] for i in range(len(domain_cnt))])

        print(f'{mode}_data: {X.shape}, {y.shape}', end='')

        X, y = X.to(self.device), y.to(self.device)
        if self.is_multi_tower:
            dataset = TensorDataset(X, y, group)
        else:
            dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, self.config.bs, shuffle=True)

        return data_loader

    def convert2domain_data_loader(self, data, cols, mode):
        cols = cols if data is None else data.columns
        x_cols = [col for col in cols if col != self.label_name]

        if os.path.exists(os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth')):
            X = torch.load(os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth')).to(torch.int)
            y = torch.load(os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth')).to(torch.short)
        else:
            X, y = self.save_tensor_from_data(data, cols, mode)

        assert X.shape[1] == len(x_cols), f'X.shape[1] != len(x_cols): {X.shape[1]} != {len(x_cols)}'

        if self.domain_filter is not None:
            mask = torch.isin(X[:, self.domain_idx], torch.tensor(self.domain_filter))
            X = X[mask]
            y = y[mask]

        domain_data_loader = []
        domain_list = self.domain_filter if self.domain_filter is not None else range(self.n_domain)
        for d in domain_list:
            if self.domain_filter is not None and d not in self.domain_filter:
                continue
            mask = (X[:, self.domain_idx] == d)
            domain_X = X[mask]
            domain_y = y[mask]
            domain_X, domain_y = domain_X.to(self.device), domain_y.to(self.device)
            domain_data_loader.append(DataLoader(TensorDataset(domain_X, domain_y), self.config.bs,
                                                 shuffle=True))
            if mode == 'train':
                self.train_domain_batch_seq.extend([d]*np.ceil(domain_X.shape[0]*1./self.config.bs).astype(int))
            elif mode == 'valid':
                self.valid_domain_batch_seq.extend([d]*np.ceil(domain_X.shape[0]*1./self.config.bs).astype(int))
            elif mode == 'test':
                self.test_domain_batch_seq.extend([d]*np.ceil(domain_X.shape[0]*1./self.config.bs).astype(int))
        if mode == 'train':
            domain_val = X[:, self.domain_idx]
            domain_cnt = domain_val.bincount()
            print(mode, 'domain data cnt: ', domain_cnt)
            self.domain_cnt_weight = np.array([domain_cnt[i]/X.shape[0] for i in range(len(domain_cnt))])
            np.random.shuffle(self.train_domain_batch_seq)  # shuffle train_domain_batch_seq
        elif mode == 'valid':
            np.random.shuffle(self.valid_domain_batch_seq)
        elif mode == 'test':
            np.random.shuffle(self.test_domain_batch_seq)

        return domain_data_loader

    def get_data(self):
        print('========Reading data========')
        cols, data = self.read_split_data(self.preprocess_path)
        print('after multi-hot features flatten')
        if 'cdc' in self.model:
            self.train_data_loader = self.convert2domain_data_loader(data[0], cols, mode='train')
            self.valid_data_loader = self.convert2domain_data_loader(data[1], cols, mode='valid')
            self.test_data_loader = self.convert2domain_data_loader(data[2], cols, mode='test')
        else:
            self.train_data_loader = self.convert2data_loader(data[0], cols, mode='train')
            self.valid_data_loader = self.convert2data_loader(data[1], cols, mode='valid')
            self.test_data_loader = self.convert2data_loader(data[2], cols, mode='test')

        print('\n========Finish reading data========')
        return self.train_data_loader, self.valid_data_loader, self.test_data_loader

    def get_model(self):
        if self.model == 'deepfm':
            assert self.config.group_strategy == 'mix', 'deepfm only support mix group strategy'
            model = DeepFM(self.feature_dims, self.embed_dim, mlp_dims=(256, 128),
                           l2_reg_embedding=self.config.l2_reg_embedding,
                           l2_reg_linear=self.config.l2_reg_dnn,
                           l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'dcn':
            assert self.config.group_strategy == 'mix', 'dcn only support mix group strategy'
            model = DCN(self.feature_dims, self.embed_dim,
                        n_cross_layers=3, mlp_dims=self.config.mlp_dims,
                        l2_reg_embedding=self.config.l2_reg_embedding,
                        l2_reg_linear=self.config.l2_reg_dnn,
                        l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'dcnv2':
            assert self.config.group_strategy == 'mix', 'dcnv2 only support mix group strategy'
            model = DCNv2(self.feature_dims, self.embed_dim,
                          n_cross_layers=3, mlp_dims=self.config.mlp_dims,
                          l2_reg_embedding=self.config.l2_reg_embedding,
                          l2_reg_linear=self.config.l2_reg_dnn,
                          l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'autoint':
            assert self.config.group_strategy == 'mix', 'autoint only support mix group strategy'
            model = AutoInt(self.feature_dims, self.embed_dim,
                            atten_embed_dim=64, mlp_dims=self.config.mlp_dims,
                            l2_reg_embedding=self.config.l2_reg_embedding,
                            l2_reg_linear=self.config.l2_reg_dnn,
                            l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'ple':
            model = PLE(self.feature_dims, self.embed_dim,
                        n_tower=self.n_tower,
                        n_expert_specific=self.config.ple_n_expert_specific,
                        n_expert_shared=self.config.ple_n_expert_shared,
                        expert_dims=self.config.ple_expert_dims,
                        tower_dims=self.config.ple_tower_dims, config=self.config,
                        l2_reg_embedding=self.config.l2_reg_embedding,
                        l2_reg_linear=self.config.l2_reg_dnn,
                        l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'mmoe':
            model = MMoE(self.feature_dims, self.embed_dim,
                         n_tower=self.n_tower, n_expert=self.config.mmoe_n_expert,
                         expert_dims=self.config.mmoe_expert_dims,
                         tower_dims=self.config.mmoe_tower_dims, config=self.config,
                         l2_reg_embedding=self.config.l2_reg_embedding,
                         l2_reg_linear=self.config.l2_reg_dnn,
                         l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'pepnet':
            model = PEPNet(self.feature_dims, self.embed_dim,
                           n_tower=self.n_tower, tower_dims=self.config.tower_dims,
                           gate_hidden_dim=self.config.gate_hidden_dim,
                           domain_idx=self.domain_idx, use_ppnet=True,
                           config=self.config,
                           l2_reg_embedding=self.config.l2_reg_embedding,
                           l2_reg_linear=self.config.l2_reg_dnn,
                           l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'epnet':
            model = PEPNet(self.feature_dims, self.embed_dim,
                           n_tower=self.n_tower, tower_dims=self.config.tower_dims,
                           gate_hidden_dim=self.config.gate_hidden_dim,
                           domain_idx=self.domain_idx, use_ppnet=False, config=self.config,
                           l2_reg_embedding=self.config.l2_reg_embedding,
                           l2_reg_linear=self.config.l2_reg_dnn,
                           l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'pepnet-single':
            model = PEPNet(self.feature_dims, self.embed_dim,
                           n_tower=1, tower_dims=self.config.tower_dims, gate_hidden_dim=self.config.gate_hidden_dim,
                           domain_idx=self.domain_idx, use_ppnet=True, config=self.config,
                           l2_reg_embedding=self.config.l2_reg_embedding,
                           l2_reg_linear=self.config.l2_reg_dnn,
                           l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'epnet-single':
            model = PEPNet(self.feature_dims, self.embed_dim,
                           n_tower=1, tower_dims=self.config.tower_dims, gate_hidden_dim=self.config.gate_hidden_dim,
                           domain_idx=self.domain_idx, use_ppnet=False, config=self.config,
                           l2_reg_embedding=self.config.l2_reg_embedding,
                           l2_reg_linear=self.config.l2_reg_dnn,
                           l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'star':
            model = STAR(self.feature_dims, self.embed_dim,
                         n_tower=self.n_tower, tower_dims=self.config.tower_dims,
                         domain_idx=self.domain_idx, config=self.config,
                         l2_reg_embedding=self.config.l2_reg_embedding,
                         l2_reg_linear=self.config.l2_reg_dnn,
                         l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'adl' or self.model == 'adl-split':
            # adl: 由n_cluster确定n_tower
            # adl-split: 由domain2group_dict确定n_tower
            model = ADL(self.feature_dims, self.embed_dim, n_tower=self.n_tower,
                        tower_dims=self.config.tower_dims, domain_idx=self.domain_idx, dlm_iters=self.config.dlm_iters,
                        device=self.device, config=self.config,
                        l2_reg_embedding=self.config.l2_reg_embedding,
                        l2_reg_linear=self.config.l2_reg_dnn,
                        l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'hinet':
            model = HiNet(self.feature_dims, self.embed_dim,
                          n_tower=self.n_tower, sei_dims=self.config.sei_dims, tower_dims=self.config.tower_dims,
                          domain_idx=self.domain_idx, device=self.device, config=self.config,
                          l2_reg_embedding=self.config.l2_reg_embedding,
                          l2_reg_linear=self.config.l2_reg_dnn,
                          l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'adasparse':
            model = AdaSparse(self.feature_dims, self.embed_dim,
                              hidden_dims=self.config.mlp_dims, domain_idx=self.domain_idx,
                              config=self.config,
                              l2_reg_embedding=self.config.l2_reg_embedding,
                              l2_reg_linear=self.config.l2_reg_dnn,
                              l2_reg_dnn=self.config.l2_reg_dnn)
        elif self.model == 'cdc':
            assert self.config.group_strategy == 'mix', 'cdc only support mix group strategy'
            model = CDC(self.feature_dims, self.embed_dim,
                             n_tower=self.n_tower,
                             n_domain=self.n_domain,
                             base_model=self.base_model,
                             expert_dims=self.config.mlp_dims,
                             tower_dims=self.config.cdc_tower_dims,
                             domain_idx=self.domain_idx,
                             domain_cnt_weight=self.domain_cnt_weight,
                             n_causal_mask=self.config.n_causal_mask,
                             use_metric=self.config.use_metric,
                             device=self.device,
                             config=self.config,
                             savefig_folder=f'{self.model}_{self.latest_model_inx + 1}',
                             l2_reg_embedding=self.config.l2_reg_embedding,
                             l2_reg_linear=self.config.l2_reg_dnn,
                             l2_reg_dnn=self.config.l2_reg_dnn)
        else:
            raise ValueError('Unknown model: ' + self.model)
        return model.to(self.device)

    def is_continuable(self, model, result_dict, epoch_i, optimizer):
        # if result_dict['total_auc'] > self.best_auc:
        if result_dict['mean_auc'] > self.best_mean_auc:  # use mean_auc to early stop
            print('use mean_auc to early stop')
            self.trial_counter = 0
            self.best_auc = result_dict['total_auc']
            self.best_loss = result_dict['total_loss']
            save_dict = {'epoch': epoch_i + 1, 'state_dict': model.state_dict(), 'best_auc': self.best_auc,
                         'best_result': result_dict, 'preprocess_path': self.preprocess_path,
                         'optimizer': optimizer.state_dict()}
            if result_dict.get('mean_auc') is not None:
                self.best_mean_auc = result_dict['mean_auc']
                self.best_mean_loss = result_dict['mean_loss']
                save_dict['best_mean_auc'] = self.best_mean_auc
                save_dict['best_mean_loss'] = self.best_mean_loss
            if 'cdc' in self.model:
                save_dict['domain2group_list'] = model.domain2group_list
                save_dict['s_group2domain_list'] = model.s_group2domain_list

            torch.save(save_dict, self.save_model_path)
            print(f'current best epoch: {epoch_i - self.trial_counter + 1}, '
                  f'auc: {self.best_auc:.4f}, loss: {self.best_loss:.4f}, '
                  f'mean_auc: {self.best_mean_auc:.4f}, mean_loss: {self.best_mean_loss:.4f}')
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

    def train(self, data_loader, model, criterion, optimizer, epoch_i):
        print('Training Epoch {}:'.format(epoch_i + 1))
        model.train()
        loss_sum = 0
        log_interval = 204800//self.config.bs
        tk0 = tqdm.tqdm(data_loader, smoothing=0, desc=f'Training Epoch {epoch_i + 1}', mininterval=30)
        for i, batch in enumerate(tk0):
            if self.is_concat_group:
                X, y, group = batch
                pred, y = model(X, group, targets=y)
                loss = criterion(pred.squeeze(), y.squeeze().float())
            elif self.is_multi_tower:
                X, y, group = batch
                pred = model(X)
                loss = criterion(pred.gather(1, group).squeeze(1), y.squeeze().float())
            else:
                X, y = batch
                pred = model(X)
                loss = criterion(pred.squeeze(), y.squeeze().float())
            loss = loss + model.get_regularization_loss(device=self.device)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=format(loss_sum / log_interval, '.4f'))
                wandb.log({'train_loss': (loss_sum / log_interval)})
                loss_sum = 0

    def get_domain_data(self, d, mode='train'):
        if isinstance(d, (int, np.integer)):
            if mode == 'train':
                try:
                    return next(self.train_data_generator[d])
                except StopIteration:
                    self.train_data_generator[d] = iter(self.train_data_loader[d])
                    return next(self.train_data_generator[d])
            elif mode == 'valid':
                try:
                    return next(self.valid_data_generator[d])
                except StopIteration:
                    self.valid_data_generator[d] = iter(self.valid_data_loader[d])
                    return next(self.valid_data_generator[d])
            elif mode == 'test':
                try:
                    return next(self.test_data_generator[d])
                except StopIteration:
                    self.test_data_generator[d] = iter(self.test_data_loader[d])
                    return next(self.test_data_generator[d])
        else:
            torch_X, torch_y = [], []
            np.random.shuffle(d)
            for d_i in d:
                tmp_X, tmp_y = self.get_domain_data(d_i, mode)
                torch_X.append(tmp_X)
                torch_y.append(tmp_y)
            return torch.cat(torch_X, dim=0), torch.cat(torch_y, dim=0)

    def update_matrix_cdc(self, model, criterion, optimizer, update_matrix_step):
        def cdc_train_update_with_domain(train_domain_i, mode, num_interval):
            if isinstance(train_domain_i, (int, np.integer)):
                train_domain_i_list = [train_domain_i] * num_interval
            else:
                train_domain_i = list(train_domain_i)
                tmp_list = train_domain_i * num_interval
                train_domain_i_list = [tmp_list[i:i + 7] for i in range(0, len(tmp_list), 7)]

            for single_train_domain in train_domain_i_list:
                domain_train_X, domain_train_y = self.get_domain_data(single_train_domain)
                if isinstance(single_train_domain, (int, np.integer)):
                    pred = model(domain_train_X, mode=mode, domain_i=single_train_domain)
                else:
                    pred = model(domain_train_X, mode=mode)
                loss = criterion(pred.squeeze(), domain_train_y.squeeze().float())
                loss = loss + model.get_regularization_loss(device=self.device)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                del domain_train_X, domain_train_y, pred, loss

        def cdc_test_all_domain():
            model.eval()
            with torch.no_grad():
                matrix_one_line = np.zeros(self.n_domain)
                for d_j in range(self.n_domain):
                    domain_train_X, domain_train_y = self.get_domain_data(d_j)
                    pred = model(domain_train_X, mode='split', domain_i=d_j)
                    matrix_one_line[d_j] = model.get_matrix_metric(pred.squeeze(), domain_train_y.squeeze().float())
            return torch.tensor(matrix_one_line, dtype=torch.float).to(self.device)

        model.save_model_state()

        # get treatment matrix
        for line_i in tqdm.tqdm(range(self.config.n_causal_mask), desc='Get causal mask', mininterval=20):
            train_domain_i = np.random.choice(range(self.n_domain),
                                              p=self.domain_cnt_weight,
                                              size=np.random.randint(5, self.n_domain))
            cdc_train_update_with_domain(train_domain_i, 'split', update_matrix_step)
            model.matrix_mask[line_i] = cdc_test_all_domain()
            model.load_model_state()

        # get matrix A
        model.matrix_A[self.n_domain] = cdc_test_all_domain()
        for d_i in tqdm.tqdm(range(self.n_domain), desc='Get matrix A', mininterval=15):
            model.train()
            cdc_train_update_with_domain(d_i, 'split', update_matrix_step)
            model.matrix_A[d_i] = cdc_test_all_domain()
            model.load_model_state()

        # get matrix B
        if max(model.domain2group_list) > 0:
            # 已经有了分组，那matrix B的全集就按照单组中所有的domain算
            tk_B = tqdm.tqdm(range(self.n_domain+self.n_cluster), desc='Get matrix B', mininterval=15)
        else:
            tk_B = tqdm.tqdm(range(self.n_domain+1), desc='Get matrix B', mininterval=15)
        for d_i in tk_B:
            if d_i >= self.n_domain:
                train_domain_i = model.domain2group_list[d_i-self.n_domain]
            else:
                train_domain_i = [d for d in model.s_group2domain_list[model.domain2group_list[d_i]] if d != d_i]
            cdc_train_update_with_domain(train_domain_i, 'split', update_matrix_step)
            model.matrix_B[d_i] = cdc_test_all_domain()
            model.load_model_state()

        self.domain2group_list = model.update_group()

    def train_cdc(self, model, criterion, optimizer, epoch_i):
        print('Training Epoch {}:'.format(epoch_i + 1))
        model.train()
        log_interval = 204800//self.config.bs

        warmup_step = max(5, (self.config.warmup_step*1024)//self.config.bs)  # 默认200
        update_matrix_step = max(1, (self.config.update_matrix_step*1024)//self.config.bs) \
            if self.config.update_matrix_step != 0 else 0  # 默认为2
        update_interval = (self.config.update_interval*1024)//self.config.bs  # 大约是1/5个epoch更新一次聚类
        print(f'warmup_step: {warmup_step}, '
              f'update_matrix_step: {update_matrix_step}, update_interval: {update_interval}')

        # warm up the model
        if epoch_i == 0:
            print('========Warm up========')
            loss_sum = 0
            tk0 = tqdm.tqdm(range(warmup_step), smoothing=0, mininterval=20)
            for i in tk0:
                d = np.random.choice(range(self.n_domain), p=self.domain_cnt_weight)  # 选择domain
                domain_train_X, domain_train_y = self.get_domain_data(d)
                pred = model(domain_train_X, mode='warmup')
                loss = criterion(pred.squeeze(), domain_train_y.squeeze().float())
                loss = loss + model.get_regularization_loss(device=self.device)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                if (i + 1) % log_interval == 0:
                    tk0.set_postfix(loss=format(loss_sum / log_interval, '.4f'))
                    wandb.log({'train_loss': (loss_sum / log_interval)})
                    loss_sum = 0
            del loss_sum

        loss_sum = 0
        tk0 = tqdm.tqdm(self.train_domain_batch_seq, smoothing=0, desc=f'Training Epoch {epoch_i + 1}', mininterval=30)
        for i, d in enumerate(tk0):
            train_X, train_y = self.get_domain_data(d)
            if (epoch_i == 0 and i == 0) or ((i+1) % update_interval == 0):
                self.update_matrix_cdc(model, criterion, optimizer, update_matrix_step)
            pred = model(train_X, mode='split', domain_i=d)
            loss = criterion(pred.squeeze(), train_y.squeeze().float())
            loss = loss + model.get_regularization_loss(device=self.device)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=format(loss_sum / log_interval, '.4f'))
                wandb.log({'train_loss': (loss_sum / log_interval)})
                loss_sum = 0

    def test(self, data_loader, model, mode='valid', cdc_final=False):
        print('Evaluating:')
        model.eval()
        targets, predicts, domains = [], [], []

        with torch.no_grad():
            if 'cdc' in self.model:
                for d in tqdm.tqdm(self.valid_domain_batch_seq if mode == 'valid' else self.test_domain_batch_seq,
                                   smoothing=0, mininterval=60):
                    X, y = self.get_domain_data(d, mode=mode)
                    pred = model(X, mode='split', domain_i=d)

                    targets.append(y.squeeze().cpu().numpy())
                    predicts.append(pred.squeeze().cpu().numpy())
                    domains.append(X[:, self.domain_idx].cpu().numpy())
            else:
                for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=60):
                    if self.is_multi_tower:
                        X, y, group = batch
                        if 'adl' in self.model:
                            pred = model(X, is_training=False)
                        else:
                            pred = model(X).gather(1, group)
                    else:
                        X, y = batch
                        pred = model(X)
                    targets.append(y.squeeze().cpu().numpy())
                    predicts.append(pred.squeeze().cpu().numpy())
                    domains.append(X[:, self.domain_idx].cpu().numpy())

        targets = np.concatenate(targets)
        predicts = np.concatenate(predicts)
        domains = np.concatenate(domains)

        result_dict = dict()
        result_dict['total_auc'] = roc_auc_score(targets, predicts)
        result_dict['total_loss'] = log_loss(targets, predicts)
        # print(type(result_dict['total_auc']), type(result_dict['total_loss']))
        if self.config.is_evaluate_multi_domain:
            result_dict.update(self.evaluate_multi_domain(targets, predicts, domains))

        return result_dict

    def evaluate_multi_domain(self, targets, predicts, domains, return_type='dict'):
        df = pd.DataFrame({'targets': targets, 'predicts': predicts, 'domains': domains})
        if return_type == 'dict':
            domain_auc, domain_loss = dict(), dict()
        else:
            domain_auc, domain_loss = np.zeros(self.n_domain), np.zeros(self.n_domain)
        mean_auc, mean_loss = 0, 0

        for domain, group in df.groupby('domains'):
            try:
                auc = roc_auc_score(group['targets'], group['predicts'])
                loss = log_loss(group['targets'], group['predicts'])
            except ValueError:
                # 处理无法计算 AUC 或 Loss 的情况（例如，某个类别的标签只有一种）
                auc, loss = np.nan, np.nan

            domain_auc[domain], domain_loss[domain] = auc, loss
            mean_auc += self.domain_cnt_weight[domain] * auc
            mean_loss += self.domain_cnt_weight[domain] * loss

        return dict({'domain_auc': domain_auc, 'domain_loss': domain_loss,
                     'mean_auc': mean_auc, 'mean_loss': mean_loss})

    def main(self):
        train_data_loader, valid_data_loader, test_data_loader = self.get_data()
        if 'cdc' in self.model and ('wo' not in self.model):
            self.train_data_generator = [iter(self.train_data_loader[d]) for d in range(self.n_domain)]
            self.valid_data_generator = [iter(self.valid_data_loader[d]) for d in range(self.n_domain)]
            self.test_data_generator = [iter(self.test_data_loader[d]) for d in range(self.n_domain)]
        model = self.get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr,
                                     betas=(0.9, 0.99), eps=1e-8, weight_decay=self.config.wd)

        criterion = torch.nn.BCELoss(reduction='mean')

        if self.config.is_increment:
            print('loading model for increment learning...')
            checkpoint = torch.load(os.path.join(self.config.save_path,
                                                 f'{self.model}_{self.latest_model_inx}.pth.tar'),
                                    map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])

        if 'cdc' in self.model:
            for epoch_i in range(self.epoch):
                self.train_cdc(model, criterion, optimizer, epoch_i)
                result_dict = self.test(valid_data_loader, model, mode='valid')
                wandb.log(result_dict)
                print(f'validation: auc: {result_dict["total_auc"]:.4f}, loss: {result_dict["total_loss"]:.4f}')
                if result_dict.get("mean_auc") is not None:
                    print(f'validation: mean_auc: {result_dict["mean_auc"]:.4f}, '
                          f'mean_loss: {result_dict["mean_loss"]:.4f}')
                if not self.is_continuable(model, result_dict, epoch_i, optimizer):
                    break
            # print(f'cdc domain2group: {model.domain2group_list}')
            wandb.log({'domain2group_list': model.domain2group_list})
            wandb.log({'s_group2domain_list': model.s_group2domain_list})
        else:
            for epoch_i in range(self.epoch):
                self.train(train_data_loader, model, criterion, optimizer, epoch_i)
                result_dict = self.test(valid_data_loader, model, mode='valid')
                wandb.log(result_dict)
                print(f'validation: auc: {result_dict["total_auc"]:.4f}, loss: {result_dict["total_loss"]:.4f}')
                if result_dict.get("mean_auc") is not None:
                    print(f'validation: mean_auc: {result_dict["mean_auc"]:.4f}, '
                          f'mean_loss: {result_dict["mean_loss"]:.4f}')
                if not self.is_continuable(model, result_dict, epoch_i, optimizer):
                    break

        print('loading best model...')
        checkpoint = torch.load(self.save_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        result_dict = self.test(test_data_loader, model, mode='test')
        wandb.log(result_dict)
        wandb.log({'epoch_i': epoch_i})
        print('test: ', list(result_dict.items()))
