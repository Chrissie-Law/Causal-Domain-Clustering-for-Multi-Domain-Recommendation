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
    """
    Parse command line arguments and set up configuration for the CDC model.

    This function handles:
    1. Parsing command line arguments for model parameters
    2. Setting seed for reproducibility
    3. Loading default configurations from config file

    Returns:
        args (argparse.Namespace): The configuration object containing all experiment parameters.
    """
    parser = argparse.ArgumentParser()
    # Basic model and dataset parameters
    parser.add_argument('--model', default='deepfm', help="Model type: 'deepfm', 'dcn', 'autoint', 'cdc', etc.")
    parser.add_argument('--dataset_name', default='amazon', help="Dataset name: 'amazon', 'aliccp'")
    parser.add_argument('--base_model', default='mmoe',
                        help="Base model for multi-domain methods: 'mmoe', 'ple', 'pepnet', 'epnet', or 'star'.")
    # Training parameters
    parser.add_argument('--seed', type=int, default=2000, help="Random seed for reproducibility")
    parser.add_argument('--is_set_seed', type=int, default=0,
                        help="Set to 1 to enable seed setting for reproducible experiments")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--bs', type=int, default=1024, help="Batch size")
    parser.add_argument('--l2_reg', type=float, default=1e-5, help="L2 regularization for DNN and embedding layers")
    parser.add_argument('--epoch', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--embed_dim', type=int, default=40, help="Embedding dimension for feature embeddings")
    parser.add_argument('--prepare2train_month', type=int, default=12, help="Months of data used for preprocessing")
    parser.add_argument("--group_strategy", default='mix', help="Domain grouping strategy, specified in config.py")
    # CDC clustering parameters
    parser.add_argument("--n_cluster", type=int, default=4,
                        help="Number of domain clusters to create")
    parser.add_argument("--update_matrix_step", type=int, default=2,
                        help="Number of steps between affinity matrix updates")
    parser.add_argument("--warmup_step", type=int, default=200,
                        help="Number of training steps for model warm-up before clustering")
    parser.add_argument("--p_weight", type=float, default=0.02,
                        help="Initial weight for the domain affiliation score in source domain selection")
    parser.add_argument("--p_weight_method", default='exponential_decay',
                        help="Method to decay p_weight: 'exponential_decay', 'linear_decay', or 'quadratic_decay'")
    parser.add_argument("--p_weight_exp_decay", type=float, default=0.4,
                        help="Decay rate for p_weight when using exponential decay")
    parser.add_argument("--n_causal_mask", type=int, default=50,
                        help="Number of random treatments for causal discovery")
    parser.add_argument("--update_interval", type=int, default=1000,
                        help="Number of steps between domain clustering updates")
    parser.add_argument("--affinity_func", type=str, default='minus',
                        help="Function to calculate domain affinity: 'minus' or 'divide'")
    parser.add_argument("--old_matrix_weight", type=float, default=0,
                        help="Weight of previous affinity matrices when updating (0-1)")
    args = parser.parse_args()

    if args.is_set_seed == 0:
        # Auto-generate a deterministic seed based on the experiment config
        args.seed = hash(frozenset(vars(args).items())) % 10000
        args.is_set_seed = 1
        print('set args.seed:', args.seed)

    # Merge additional defaults from config.py
    for key, value in vars(config).items():
        if key not in vars(args) and not key.startswith('__'):
            setattr(args, key, value)
    setattr(args, 'l2_reg_embedding', args.l2_reg)
    setattr(args, 'l2_reg_linear', args.l2_reg)
    setattr(args, 'l2_reg_dnn', args.l2_reg)

    # Set random seeds (for reproducibility)
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
    config = load_config()  # Load all training arguments and config

    datapre = DataPreprocessing(
        config.data_path,
        dataset_name=config.dataset_name,
        domains=[],
        prepare2train_month=config.prepare2train_month
    )
    datapre.main()  # Run the preprocessing pipeline
    datapre.update_config(config)  # Update config with preprocessing results

    # Set up wandb for experiment tracking
    os.environ['WANDB_CACHE_DIR'] = os.path.join('wandb', 'cache')
    wandb.init(project="cdc", entity="anonymous", config=config)
    print('config:', type(config), config.__dict__)

    print('============Model Training============')
    print(f'model:{config.model}, lr:{config.lr}, bs:{config.bs}, ',
          f'embed_dim:{config.embed_dim}, gpu:{config.gpu}, epoch:{config.epoch}, '
          f'seed:{config.seed if config.is_set_seed else None}, '
          f'dataset_name:{config.dataset_name}, strategy:{config.group_strategy}')
    Run(config).main()
