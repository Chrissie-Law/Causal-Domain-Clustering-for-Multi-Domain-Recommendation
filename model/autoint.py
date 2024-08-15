#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from model.layer import BaseModel, MultiLayerPerceptron


class AutoInt(BaseModel):
    """
    AutoInt Network architecture.

    Reference: Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, feature_dims, embed_dim, atten_embed_dim=None, att_layer_num=3,
                 att_head_num=2, att_res=True, mlp_dims=(256, 128), dropout=0.2,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5):

        super(AutoInt, self).__init__(feature_dims, embed_dim,
                                      l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = 'autoint'

        if len(mlp_dims) <= 0 and att_layer_num <= 0:
            raise ValueError("Either MLP hidden_layer or att_layer_num must > 0")

        if atten_embed_dim is None:
            atten_embed_dim = embed_dim
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.atten_output_dim = self.embedding.output_dim0 * atten_embed_dim
        self.att_res = att_res
        self.dnn = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, att_head_num, dropout=dropout) for _ in range(att_layer_num)
        ])

        if self.att_res:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

        final_dim = mlp_dims[-1] + self.atten_output_dim
        self.dnn_linear = nn.Linear(final_dim, 1, bias=False)
        self.output_layer = nn.Sigmoid()

        self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)

    def forward(self, x):
        embed_x = self.embedding(x)  # size: batch_size, num_fields, embed_dim
        atten_x = self.atten_embedding(embed_x)  # size: batch_size, num_fields, atten_embed_dim
        cross_term = atten_x.transpose(0, 1)  # size: num_fields, batch_size, atten_embed_dim
        for self_attn in self.self_attns:
            # input size of MultiheadAttention: L(sequence length), N(batch size), E(embedding dimension)
            # input: query, key, value. output: attn_output(same size of input), attn_output_weights(L*L)
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)  # batch_size, num_fields, atten_embed_dim
        if self.att_res:
            V_res = self.V_res_embedding(embed_x)
            cross_term += V_res
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)  # transpose-contiguous-view

        mlp_input = torch.flatten(embed_x, start_dim=1)
        final_out = torch.cat((cross_term, self.dnn(mlp_input)), dim=1)
        y_pred = self.output_layer(self.dnn_linear(final_out)+self.linear(mlp_input))
        return y_pred.squeeze(1)