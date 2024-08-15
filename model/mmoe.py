#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from model.layer import BaseModel, MultiLayerPerceptron, CrossNetwork


class MMoE(BaseModel):
    """
    Multi-gate Mixture-of-Experts model. MMoE与DCN concat后过linear层，在与线性输出相加，效果较好，即is_concat_linear_cn=False.
    Reference:
    Jiaqi Ma, et al. MMoE:
    Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts, 2018.
    """

    def __init__(self, feature_dims, embed_dim, n_tower, n_expert,
                 expert_dims, tower_dims, dropout=0.2, config=None,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5,
                 model_name='mmoe'):
        super(MMoE, self).__init__(feature_dims, embed_dim,
                                   l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.config = config
        self.model_name = model_name
        self.n_tower = n_tower
        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)

        if self.use_dcn:
            self.cn = CrossNetwork(self.embed_output_dim, config.n_cross_layers)
        if self.use_atten:
            self.build_atten(config, dropout)

        self.experts = nn.ModuleList(MultiLayerPerceptron(self.embed_output_dim, expert_dims, dropout,
                                                          output_layer=False) for _ in range(n_expert))
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(self.embed_output_dim, n_expert),
                          nn.Softmax(dim=1))
            for _ in range(self.n_tower)])

        self.towers, self.towers_linear, self.output_layers = self.build_tower_output(n_tower, expert_dims[-1],
                                                                                      tower_dims, dropout)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.experts.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.towers.named_parameters()), l2=l2_reg_dnn)
        if self.use_dcn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)

    def forward(self, x):
        embed_x = self.embedding(x, squeeze_dim=True)  # [batch_size, input_dims]

        expert_outs = [expert(embed_x).unsqueeze(1)
                       for expert in self.experts]  # expert_out[i]: [batch_size, 1, expert_dims[-1]]
        expert_outs = torch.cat(expert_outs, dim=1)  # [batch_size, n_expert, expert_dims[-1]]
        gate_outs = [gate(embed_x).unsqueeze(-1) for gate in self.gates]  #gate_out[i]: [batch_size, n_expert, 1]
        tower_inputs = [torch.sum(torch.mul(gate_out, expert_outs), dim=1) for gate_out in gate_outs]
        # expert_weight: [batch_size, n_expert, expert_dims[-1]], expert_pooling: [batch_size, expert_dims[-1]]

        # predict
        other_outs = [self.linear(embed_x)]
        if self.use_dcn:
            cn_out = self.cn(embed_x)
            other_outs.append(cn_out)
        if self.use_atten:
            atten_out = self.atten_forward(embed_x)
            other_outs.append(atten_out)

        y = self.tower_forward(tower_inputs, other_outs)

        return y
