#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.layer import BaseModel, CrossNetwork, MultiLayerPerceptron


class SEI(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], expert_num=4, dropout=0.2):
        super().__init__()
        self.expert_num = expert_num

        self.experts = nn.ModuleList([MultiLayerPerceptron(input_dim, hidden_dims, dropout, output_layer=False) for _ in range(expert_num)])
        self.gate = nn.Sequential(nn.Linear(input_dim, self.expert_num), torch.nn.Softmax(dim=1))

    def forward(self, X):
        fea = torch.cat([self.experts[i](X).unsqueeze(1) for i in range(self.expert_num)], dim=1)
        gate_value = self.gate(X).unsqueeze(1)
        out = torch.bmm(gate_value, fea).squeeze(1)

        return out


class HiNet(BaseModel):
    def __init__(self, feature_dims, embed_dim=10, n_tower=6, sei_dims=None, tower_dims=None,
                 domain_idx=None, device='cpu', dropout=0.2, config=None,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5):
        super(HiNet, self).__init__(feature_dims, embed_dim,
                                    l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)

        self.n_tower = n_tower
        self.device = device
        self.domain_idx = domain_idx
        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)

        if self.use_dcn:
            self.cn = CrossNetwork(self.embed_output_dim, config.n_cross_layers)
        if self.use_atten:
            self.build_atten(config, dropout)

        self.specific_seis = nn.ModuleList([SEI(self.embed_output_dim, hidden_dims=sei_dims, dropout=dropout)
                                            for _ in range(self.n_tower)])
        self.shared_seis = SEI(self.embed_output_dim, hidden_dims=sei_dims, dropout=dropout)
        self.san_gate = torch.nn.Sequential(torch.nn.Linear(embed_dim, n_tower), torch.nn.Softmax(dim=1))
        self.tower = MultiLayerPerceptron(sei_dims[-1]*3, tower_dims, dropout, output_layer=False)
        self.tower_linear = nn.Linear(tower_dims[-1], 1, bias=False)
        self.output_layer = nn.Sigmoid()

        if self.use_dcn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.specific_seis.named_parameters()),
            l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.shared_seis.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.san_gate.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower.named_parameters()), l2=l2_reg_dnn)

    def forward(self, x, x_group, targets=None):
        embed_x = self.embedding(x, squeeze_dim=False)
        domain_embed = embed_x[:, self.domain_idx, :].squeeze(1)
        embed_x = embed_x.flatten(start_dim=1)

        specific_feas = [self.specific_seis[i](embed_x) for i in range(self.n_tower)]
        shared_feas = self.shared_seis(embed_x)
        san_gate_value = self.san_gate(domain_embed).unsqueeze(1)
        domain_feas = torch.stack(specific_feas, dim=1)
        san_feas = torch.bmm(san_gate_value, domain_feas).squeeze(1)

        con_feas = torch.zeros_like(specific_feas[0], dtype=torch.float, device=self.device)
        for group_id in range(self.n_tower):
            domain_mask = (x_group == group_id).squeeze()
            con_feas[domain_mask] = specific_feas[group_id][domain_mask]
        feature = torch.cat([shared_feas, con_feas, san_feas], dim=1)
        y_logits = self.tower_linear(self.tower(feature).squeeze(1))

        # other predict
        other_outs = [self.linear(embed_x)]
        if self.use_dcn:
            cn_out = self.cn(embed_x)
            other_outs.append(cn_out)
        if self.use_atten:
            atten_out = self.atten_forward(embed_x)
            other_outs.append(atten_out)
        for other in other_outs:
            y_logits += other
        pred = self.output_layer(y_logits).squeeze(1)

        return pred, targets