#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn, flatten
from model.layer import BaseModel, FeaturesLinear, MultiLayerPerceptron
from model.layer import FactorizationMachine


class DeepFM(BaseModel):
    """
    A pytorch implementation of DeepFM.

    Reference:
        Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017.
        DeepFM: a factorization-machine based neural network for CTR prediction.
        In Proceedings of the 26th International Joint Conference on Artificial Intelligence. 1725–1731.
    """

    def __init__(self, feature_dims, embed_dim, mlp_dims, dropout=0.2,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5):
        super(DeepFM, self).__init__(feature_dims, embed_dim,
                                     l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = 'deepfm'

        self.fm = FactorizationMachine(reduce_sum=True)  # 二阶
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=True)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.mlp.named_parameters()), l2=l2_reg_dnn)

        self.output_layer = nn.Sigmoid()

    def forward(self, x):
        embed_x = self.embedding(x)
        mlp_input = flatten(embed_x, start_dim=1)

        y = self.output_layer(self.linear(mlp_input) + self.fm(embed_x) + self.mlp(mlp_input))
        return y.squeeze(1)
