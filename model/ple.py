#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from model.layer import BaseModel, MultiLayerPerceptron, CrossNetwork


class PLE(BaseModel):
    """
    Progressive Layered Extraction model. PLE无论如何concat DCN效果都会下降，因此不加入DCN
    Reference: Hongyan Tang, et al. PLE: Progressive Layered Extraction (PLE):
        A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations, 2020.
    """

    def __init__(self, feature_dims, embed_dim, n_tower,
                 n_expert_specific, n_expert_shared, expert_dims, tower_dims, dropout=0.2, config=None,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5, model_name='ple'):
        super(PLE, self).__init__(feature_dims, embed_dim,
                                  l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = model_name

        self.n_level = len(expert_dims)
        self.n_tower = n_tower
        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)


        if self.use_dcn:
            self.cn = CrossNetwork(self.embed_output_dim, config.n_cross_layers)
        if self.use_atten:
            self.build_atten(config, dropout)

        self.cgc_layers = nn.ModuleList(
            CGC(i + 1, self.n_level, n_tower, n_expert_specific, n_expert_shared,
                self.embed_output_dim if i == 0 else expert_dims[i-1][-1], expert_dims[i], dropout)
            for i in range(self.n_level))

        self.towers, self.towers_linear, self.output_layers = self.build_tower_output(n_tower, expert_dims[-1][-1],
                                                                                      tower_dims, dropout)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cgc_layers.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.towers.named_parameters()), l2=l2_reg_dnn)
        if self.use_dcn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)

    def forward(self, x):
        embed_x = self.embedding(x, squeeze_dim=True)  #[batch_size, input_dims]
        # cn_out = self.cn(embed_x)

        ple_inputs = [embed_x] * (self.n_tower + 1)
        ple_outs = []
        for i in range(self.n_level):
            ple_outs = self.cgc_layers[i](ple_inputs)  #ple_outs[i]: [batch_size, expert_dims[-1]]
            ple_inputs = ple_outs

        # predict
        other_outs = [self.linear(embed_x)]
        if self.use_dcn:
            cn_out = self.cn(embed_x)
            other_outs.append(cn_out)
        if self.use_atten:
            atten_out = self.atten_forward(embed_x)
            other_outs.append(atten_out)

        y = self.tower_forward(ple_outs, other_outs)
        return y


class CGC(nn.Module):
    def __init__(self, cur_level, n_level, n_task, n_expert_specific, n_expert_shared,
                 input_dims, expert_dims, dropout=0.2):
        super().__init__()
        self.cur_level = cur_level  # the CGC level of PLE
        self.n_level = n_level
        self.n_task = n_task
        self.n_expert_specific = n_expert_specific
        self.n_expert_shared = n_expert_shared
        self.n_expert_all = n_expert_specific * self.n_task + n_expert_shared
        self.experts_specific = nn.ModuleList(
            MultiLayerPerceptron(input_dims, expert_dims, dropout, output_layer=False, bn=False)
            for _ in range(self.n_task * self.n_expert_specific))
        self.experts_shared = nn.ModuleList(
            MultiLayerPerceptron(input_dims, expert_dims, dropout, output_layer=False, bn=False)
            for _ in range(self.n_expert_shared))
        self.gates_specific = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dims, self.n_expert_specific + self.n_expert_shared),
                          nn.Softmax(dim=1)) for _ in range(self.n_task)])  # n_gate_specific = n_task
        if cur_level < n_level:
            self.gate_shared = nn.Sequential(nn.Linear(input_dims, self.n_expert_all),
                                             nn.Softmax(dim=1))  # n_gate_specific = n_task

    def forward(self, x_list):
        expert_specific_outs = []  # expert_out[i]: [batch_size, 1, expert_dims[-1]]
        for i in range(self.n_task):
            expert_specific_outs.extend([
                expert(x_list[i]).unsqueeze(1)
                for expert in self.experts_specific[i * self.n_expert_specific:(i + 1) * self.n_expert_specific]
            ])
        # x_list[-1]: the input for shared experts
        expert_shared_outs = [expert(x_list[-1]).unsqueeze(1) for expert in self.experts_shared]
        # gate_out[i]: [batch_size, n_expert_specific+n_expert_shared, 1]
        gate_specific_outs = [gate(x_list[i]).unsqueeze(-1) for i, gate in enumerate(self.gates_specific)]
        cgc_outs = []
        for i, gate_out in enumerate(gate_specific_outs):
            cur_expert_list = expert_specific_outs[i * self.n_expert_specific:(i + 1) *
                                                   self.n_expert_specific] + expert_shared_outs
            expert_concat = torch.cat(cur_expert_list,
                                      dim=1)  # [batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            expert_weight = torch.mul(gate_out,
                                      expert_concat)  #[batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  # [batch_size, expert_dims[-1]]
            cgc_outs.append(expert_pooling)  # length: n_task
        if self.cur_level < self.n_level:  #not the last layer
            gate_shared_out = self.gate_shared(x_list[-1]).unsqueeze(-1)  #[batch_size, n_expert_all, 1]
            expert_concat = torch.cat(expert_specific_outs + expert_shared_outs,
                                      dim=1)  #[batch_size, n_expert_all, expert_dims[-1]]
            expert_weight = torch.mul(gate_shared_out, expert_concat)  #[batch_size, n_expert_all, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  #[batch_size, expert_dims[-1]]
            cgc_outs.append(expert_pooling)  #length: n_task+1

        return cgc_outs