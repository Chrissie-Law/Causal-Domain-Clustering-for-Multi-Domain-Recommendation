#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import random
import argparse
import config
from preprocess import DataPreprocessing
from run import Run
import wandb


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepfm')
    parser.add_argument('--dataset_name', default='amazon')
    parser.add_argument('--base_model', default='mmoe')
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--is_set_seed', type=int, default=0)  # 为了可复现实验结果，一般设置为1
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--l2_reg', type=float, default=1e-5)  # dnn层，embedding层的l2正则化
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--embed_dim', type=int, default=40)
    parser.add_argument('--run_cnt', type=int, default=0)
    parser.add_argument('--prepare2train_month', type=int, default=12)
    parser.add_argument("--group_strategy", default='mix')
    # for cdc
    parser.add_argument("--n_cluster", type=int, default=4)
    parser.add_argument("--update_matrix_step", type=int, default=2)
    parser.add_argument("--warmup_step", type=int, default=200)
    parser.add_argument("--p_weight", type=float, default=0.02)
    parser.add_argument("--p_weight_method", default='exponential_decay')
    parser.add_argument("--p_weight_exp_decay", type=float, default=0.4)
    parser.add_argument("--n_causal_mask", type=int, default=50)
    parser.add_argument("--update_interval", type=int, default=1000)
    parser.add_argument("--affinity_func", type=str, default='minus')
    parser.add_argument("--old_matrix_weight", type=float, default=0)
    args = parser.parse_args()

    if args.is_set_seed == 0:
        # 根据所有的args参数生成一个唯一的seed
        args.seed = hash(frozenset(vars(args).items())) % 10000
        args.is_set_seed = 1
        print('set args.seed:', args.seed)

    for key, value in vars(config).items():
        if key not in vars(args) and not key.startswith('__'):
            setattr(args, key, value)
    setattr(args, 'l2_reg_embedding', args.l2_reg)
    setattr(args, 'l2_reg_linear', args.l2_reg)
    setattr(args, 'l2_reg_dnn', args.l2_reg)

    if args.is_set_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    args.data_path = os.path.join(args.data_path, args.dataset_name)
    args.save_path = os.path.join(args.save_path, args.dataset_name)

    return args


if __name__ == '__main__':
    config = load_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    datapre = DataPreprocessing(config.data_path, dataset_name=config.dataset_name, domains=[],
                                prepare2train_month=config.prepare2train_month)
    datapre.main()
    datapre.update_config(config)

    os.environ['WANDB_CACHE_DIR'] = os.path.join('wandb', 'cache')
    wandb.init(project="cdc", entity="anonymous", config=config)
    print('config:', type(config), config.__dict__)

    print('============Model Training============')
    print(f'model:{config.model}, lr:{config.lr}, bs:{config.bs}, ',
          f'embed_dim:{config.embed_dim}, gpu:{config.gpu}, epoch:{config.epoch}, '
          f'seed:{config.seed if config.is_set_seed else None}, '
          f'dataset_name:{config.dataset_name}, strategy:{config.group_strategy}')
    Run(config).main()
