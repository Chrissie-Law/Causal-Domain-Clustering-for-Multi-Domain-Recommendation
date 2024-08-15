#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layer import BaseModel, CrossNetwork


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


class DNN_w_Pruner(nn.Module):
    def __init__(self, inputs_dim, hidden_units, domain_emb_dim,
                 init_std=0.0001, use_bn=False, dropout_rate=0):
        super(DNN_w_Pruner, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.pruners = nn.ModuleList(
            [nn.Linear(hidden_units[i]+domain_emb_dim, hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [nn.ReLU() for _ in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.alpha = 1
        self.beta = 2.0
        self.epsilon = 0.25

    def forward(self, inputs, domain_embs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)
            pi = self.beta * F.sigmoid(self.alpha * self.pruners[i](torch.cat([deep_input, domain_embs], dim=1)))

            pi[(pi.abs() - self.epsilon) <= 0] = 0.0
            fc = fc * pi

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class AdaSparse(BaseModel):
    def __init__(self, feature_dims, embed_dim, hidden_dims, domain_idx=None,
                 dropout=0.2, config=None,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5):
        super(AdaSparse, self).__init__(feature_dims, embed_dim,
                                        l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)

        self.domain_idx = domain_idx
        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)
        if self.use_dcn:
            self.cn = CrossNetwork(self.embed_output_dim, config.n_cross_layers)
        if self.use_atten:
            self.build_atten(config, dropout)

        self.dnn = DNN_w_Pruner(self.embed_output_dim, hidden_dims, domain_emb_dim=embed_dim,
                                use_bn=True, dropout_rate=dropout)
        self.dnn_linear = nn.Linear(hidden_dims[-1], 1)
        self.output_layer = nn.Sigmoid()

        if self.use_dcn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)

    def forward(self, x):
        embed_x = self.embedding(x, squeeze_dim=False)
        domain_embed = embed_x[:, self.domain_idx, :].squeeze(1).detach()
        embed_x = embed_x.flatten(start_dim=1)
        
        dnn_output = self.dnn(embed_x, domain_embed)

        # linear, cn, atten
        other_outs = [self.linear(embed_x)]
        if self.use_dcn:
            cn_out = self.cn(embed_x)
            other_outs.append(cn_out)
        if self.use_atten:
            atten_out = self.atten_forward(embed_x)
            other_outs.append(atten_out)

        dnn_logit = self.dnn_linear(dnn_output)
        for other in other_outs:
            dnn_logit += other
        pred = self.output_layer(dnn_logit).squeeze(1)

        return pred
