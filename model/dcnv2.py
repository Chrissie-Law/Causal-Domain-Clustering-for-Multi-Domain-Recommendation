#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from model.layer import BaseModel, MultiLayerPerceptron, CrossNetV2, CrossNetMix


class DCNv2(BaseModel):
    """
    Deep & Cross Network with a mixture of low-rank architecture

    Reference: Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020. (https://arxiv.org/abs/2008.13535)


    Args:
        features (list[Feature Class]): training by the whole module.
        n_cross_layers (int) : the number of layers of feature intersection layers
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        use_low_rank_mixture (bool): True, whether to use a mixture of low-rank architecture
        low_rank (int): the rank size of low-rank matrices
        num_experts (int): the number of expert networks
    """

    def __init__(self, feature_dims, embed_dim, n_cross_layers, mlp_dims, dropout=0.2,
                 model_structure="parallel", use_low_rank_mixture=True, low_rank=32, num_experts=4,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5):
        super(DCNv2, self).__init__(feature_dims, embed_dim,
                                    l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = 'dcnv2'

        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel"], \
               "model_structure={} not supported!".format(self.model_structure)
        if use_low_rank_mixture:
            self.crossnet = CrossNetMix(self.embed_output_dim, n_cross_layers,
                                        low_rank=low_rank, num_experts=num_experts)
        else:
            self.crossnet = CrossNetV2(self.embed_output_dim, n_cross_layers)
        if self.model_structure == "stacked":
            self.dnn = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
            final_dim = mlp_dims[-1]
        elif self.model_structure == "parallel":
            self.dnn = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
            final_dim = mlp_dims[-1] + self.embed_output_dim
        elif self.model_structure == "crossnet_only":  # only CrossNet
            final_dim = self.embed_output_dim

        self.dnn_linear = nn.Linear(final_dim, 1, bias=False)
        self.output_layer = nn.Sigmoid()

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_linear)
        regularization_modules = [self.crossnet.u_list, self.crossnet.v_list, self.crossnet.c_list]
        for module in regularization_modules:
            self.add_regularization_weight(module, l2=l2_reg_cross)

    def forward(self, x):
        embed_x = self.embedding(x, squeeze_dim=True)
        cross_out = self.crossnet(embed_x)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.dnn(embed_x)
            final_out = torch.cat([cross_out, dnn_out], dim=1)
        y_pred = self.output_layer(self.dnn_linear(final_out) + self.linear(embed_x))
        return y_pred.squeeze(1)
