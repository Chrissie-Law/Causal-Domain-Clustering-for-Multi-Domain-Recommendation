#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from model.layer import BaseModel, MultiLayerPerceptron, CrossNetwork


class PEPNet(BaseModel):
    """
    PEPNet based on DCN or AutoInt.
    Reference: Chang J, Zhang C, Hui Y, et al.
    Pepnet: Parameter and embedding personalized network for infusing with personalized prior information[C]
    //Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023: 3795-3804.
    """
    def __init__(self, feature_dims, embed_dim, n_tower, tower_dims, gate_hidden_dim=64,
                 domain_idx=None, use_ppnet=True, dropout=0.2, config=None,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5):
        super(PEPNet, self).__init__(feature_dims, embed_dim,
                                     l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        if use_ppnet:
            self.model_name = 'pepnet' if n_tower > 1 else 'pepnet-single'
        else:
            self.model_name = 'epnet' if n_tower > 1 else 'epnet-single'
        self.n_tower = n_tower
        self.domain_idx = domain_idx
        self.use_ppnet = use_ppnet
        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)

        if self.use_dcn:
            self.cn = CrossNetwork(self.embed_output_dim, config.n_cross_layers)
        if self.use_atten:
            self.build_atten(config, dropout)
        self.epnet = GateNN(self.embed_output_dim+embed_dim, gate_hidden_dim, self.embed_output_dim, dropout=dropout)
        if use_ppnet:
            self.ppnet = PPNetBlock(input_dim=self.embed_output_dim,
                                    gate_input_dim=self.embed_output_dim,
                                    tower_dims=tower_dims,
                                    gate_hidden_dim=gate_hidden_dim,
                                    n_tower=n_tower,
                                    dropout=dropout,
                                    output_layer=False)
            self.ppnet_linears = nn.ModuleList([
                nn.Linear(tower_dims[-1], 1, bias=False)
                for _ in range(n_tower)])
            self.output_layers = nn.ModuleList([nn.Sigmoid() for _ in range(n_tower)])
        else:
            if n_tower > 1:
                self.towers = nn.ModuleList(
                    MultiLayerPerceptron(self.embed_output_dim, tower_dims, dropout, output_layer=False)
                    for _ in range(self.n_tower))
                self.ppnet_linears = nn.ModuleList([
                    nn.Linear(tower_dims[-1], 1, bias=False)
                    for _ in range(n_tower)])
                self.output_layers = nn.ModuleList([nn.Sigmoid() for _ in range(n_tower)])
            else:
                self.towers = MultiLayerPerceptron(self.embed_output_dim, tower_dims, dropout, output_layer=False)
                self.ppnet_linears = nn.Linear(tower_dims[-1], 1, bias=False)
                self.output_layers = nn.Sigmoid()

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.epnet.named_parameters()), l2=l2_reg_dnn)
        if self.use_dcn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)
        if use_ppnet:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.ppnet.named_parameters()), l2=l2_reg_dnn)
        else:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.towers.named_parameters()), l2=l2_reg_dnn)

    def forward(self, x):
        embed_x = self.embedding(x, squeeze_dim=False)
        domain_embed = embed_x[:, self.domain_idx, :].squeeze(1)
        embed_x = embed_x.flatten(start_dim=1)

        epnet_weight = self.epnet(torch.cat([embed_x.detach(), domain_embed], dim=-1))
        epnet_out = embed_x * epnet_weight

        # linear, cn, atten
        other_outs = [self.linear(embed_x)]
        if self.use_dcn:
            cn_out = self.cn(embed_x)
            other_outs.append(cn_out)
        if self.use_atten:
            atten_out = self.atten_forward(embed_x)
            other_outs.append(atten_out)

        ys = []
        if self.use_ppnet:
            ppouts = self.ppnet(embed_x, epnet_out)
            for ppout, ppnet_linear, predict_layer in zip(ppouts, self.ppnet_linears, self.output_layers):
                y_logits = ppnet_linear(ppout)
                for other in other_outs:
                    y_logits += other
                ys.append(predict_layer(y_logits))
            pred = torch.cat(ys, dim=1)
        else:
            if self.n_tower > 1:
                for tower, ppnet_linear, predict_layer in zip(self.towers, self.ppnet_linears, self.output_layers):
                    y_logits = ppnet_linear(tower(epnet_out))
                    for other in other_outs:
                        y_logits += other
                    ys.append(predict_layer(y_logits))
                pred = torch.cat(ys, dim=1)
            else:
                y_logits = self.ppnet_linears(self.towers(epnet_out))
                for other in other_outs:
                    y_logits += other
                pred = self.output_layers(y_logits).squeeze(1)

        return pred


class GateNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None,
                 dropout=0.0, batch_norm=False):
        super(GateNN, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        gate_layers = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            gate_layers.append(nn.BatchNorm1d(hidden_dim))
        gate_layers.append(torch.nn.ReLU())
        if dropout > 0:
            gate_layers.append(torch.nn.Dropout(p=dropout))
        gate_layers.append(nn.Linear(hidden_dim, output_dim))
        gate_layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*gate_layers)

    def forward(self, inputs):
        return self.gate(inputs) * 2


class PPNetBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 gate_input_dim,
                 tower_dims,
                 gate_hidden_dim=None,
                 n_tower=None,
                 dropout=0.0,
                 output_layer=False):
        super(PPNetBlock, self).__init__()
        self.n_tower = n_tower
        self.n_layer = len(tower_dims)
        self.gate_layers = nn.ModuleList()
        self.tower_layers = nn.ModuleList()

        tower_dims = (input_dim,) + tower_dims
        for idx in range(self.n_layer):
            dense_layers = []
            dense_layers.append(nn.Linear(tower_dims[idx], tower_dims[idx + 1]))
            dense_layers.append(nn.BatchNorm1d(tower_dims[idx + 1]))
            dense_layers.append(torch.nn.ReLU())
            if dropout > 0:
                dense_layers.append(nn.Dropout(p=dropout))
            one_tower_layer = nn.Sequential(*dense_layers)
            self.tower_layers.append(nn.ModuleList([one_tower_layer] * self.n_tower))
            self.gate_layers.append(GateNN(gate_input_dim + input_dim, gate_hidden_dim,
                                           output_dim=tower_dims[idx]*self.n_tower))

        if output_layer:
            self.tower_layers.append(nn.ModuleList([nn.Linear(tower_dims[-1], 1)] * self.n_tower))
            self.gate_layers.append(GateNN(gate_input_dim + input_dim, gate_hidden_dim,
                                           output_dim=tower_dims[-1]*self.n_tower))

    def forward(self, feature_emb, gate_emb):
        tower_layer_inputs = [feature_emb] * self.n_tower
        gate_input = torch.cat([feature_emb.detach(), gate_emb], dim=-1)

        for gate_layer, tower_layer in zip(self.gate_layers, self.tower_layers):
            gw = gate_layer(gate_input)
            gws_split = torch.chunk(gw, self.n_tower, dim=1)
            tower_layer_outs = [tower_layer[i](tower_layer_inputs[i]*gws_split[i]) for i in range(self.n_tower)]
            tower_layer_inputs = tower_layer_outs
        return tower_layer_outs

