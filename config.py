#!/usr/bin/env python
# -*- coding: utf-8 -*-
use_cuda = 1
gpu = 0
data_path = "dataset"
save_path = "save"
itemid_all = 1368287
seq_maxlen = 5
early_stop = 2
is_increment = 0
is_evaluate_multi_domain = 1
embed_dim = 16
bs = 512
epoch = 10
wd = 1e-8

# common used mlp_dims for dcn, dcnv2, autoint
mlp_dims = (256, 128, 64)

# common used tower_dims for pepnet, epnet, epnet-single, star, adl
tower_dims = (256, 128, 64, 32)

# autoint
use_atten = True
atten_embed_dim = 64
att_layer_num = 3
att_head_num = 2
att_res = True

# dcn & dcnv2
n_cross_layers = 3

# mmoe
mmoe_n_expert = 4
mmoe_expert_dims = (256, 128, 64)
mmoe_tower_dims = (64, 32)

# ple
ple_n_expert_specific = 2
ple_n_expert_shared = 2
ple_expert_dims = ((256, 128), (64,))
ple_tower_dims = (64, 32)

# pepnet
gate_hidden_dim = 64

# hinet
sei_dims = [64, 32]


# adl
dlm_iters = 3

# cdc
n_causal_mask = 50
use_metric = 'loss'
warmup_step = 200
update_matrix_step = 2
update_interval = 1000
cdc_tower_dims = (64, 32)

domain2group_org_dict = {
    "amazon": {
        "mix": [0] * 25,
        "split": list(range(25)),
    },
    "aliccp": {
        "mix": [0] * 50,
        "split": list(range(50)),
    },
}
