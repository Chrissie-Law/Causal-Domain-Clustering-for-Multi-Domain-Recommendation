#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import Tensor
from itertools import chain
import torch.nn.functional as F
from model.layer import BaseModel, MultiLayerPerceptron, CrossNetwork


class DCN(BaseModel):
    """
    A pytorch implementation of DCN and DCN-V2.

    Reference:
        Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, feature_dims, embed_dim, n_cross_layers, mlp_dims, dropout=0.2,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5):
        super(DCN, self).__init__(feature_dims, embed_dim,
                                  l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = 'dcn'

        self.cn = CrossNetwork(self.embed_output_dim, n_cross_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.mlp_linear = nn.Linear(self.embed_output_dim + mlp_dims[-1], 1, bias=False)
        self.output_layer = nn.Sigmoid()

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.mlp.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)

    def forward(self, x):
        embed_x = self.embedding(x, squeeze_dim=True)
        cn_out = self.cn(embed_x)
        mlp_out = self.mlp(embed_x)

        x_stack = torch.cat([cn_out, mlp_out], dim=1)
        y = self.output_layer(self.linear(embed_x)+self.mlp_linear(x_stack))
        return y.squeeze(1)
