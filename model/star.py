#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor
from model.layer import BaseModel, CrossNetwork, DNN
from torch.nn.modules.batchnorm import _NormBase
import torch.nn.functional as F


class STAR(BaseModel):
    """
    STAR based on DCN.
    Reference: Sheng X R, Zhao L, Zhou G, et al.
    One model to serve all: Star topology adaptive recommender for multi-domain ctr prediction[C]
    //Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021: 4104-4113.

    """

    def __init__(self, feature_dims, embed_dim,
                 n_tower, tower_dims, domain_idx=None, dropout=0.2, config=None,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5, device=None):
        super(STAR, self).__init__(feature_dims, embed_dim,
                                   l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = 'star'
        self.n_tower = n_tower
        self.domain_idx = domain_idx
        self.device = device
        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)

        if self.use_dcn:
            self.cn = CrossNetwork(self.embed_output_dim, config.n_cross_layers)
        if self.use_atten:
            self.build_atten(config, dropout)

        self.shared_bn_weight = nn.Parameter(torch.ones(self.embed_output_dim))
        self.shared_bn_bias = nn.Parameter(torch.zeros(self.embed_output_dim))

        self.domain_norm = nn.ModuleList([MDR_BatchNorm(self.embed_output_dim) for _ in range(n_tower)])

        self.domain_dnns = nn.ModuleList([
            DNN(self.embed_output_dim, tower_dims, dropout_rate=dropout)
            for _ in range(n_tower)])

        self.domain_dnn_linears = nn.ModuleList([nn.Linear(tower_dims[-1], 1)
                                                 for _ in range(n_tower)])

        self.shared_dnn = DNN(self.embed_output_dim, tower_dims, dropout_rate=dropout)
        self.shared_dnn_linear = nn.Linear(tower_dims[-1], 1)
        self.output_layers = nn.ModuleList([nn.Sigmoid() for _ in range(n_tower)])

        if self.use_dcn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.domain_dnns.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.shared_dnn.named_parameters()), l2=l2_reg_dnn)

    def forward(self, x, x_group=None, targets=None):
        embed_x = self.embedding(x, squeeze_dim=True)

        # other predict
        other_outs = [self.linear(embed_x)]
        if self.use_dcn:
            cn_out = self.cn(embed_x)
            other_outs.append(cn_out)
        if self.use_atten:
            atten_out = self.atten_forward(embed_x)
            other_outs.append(atten_out)

        ys = []
        grouped_targets = []
        for group_id in range(self.n_tower):
            domain_dnn = self.domain_dnns[group_id]
            domain_dnn_linear = self.domain_dnn_linears[group_id]
            domain_bn = self.domain_norm[group_id]

            if x_group is None:
                domain_dnn_input, domain_label = embed_x, targets
            else:
                mask = (x_group == group_id).squeeze()
                domain_dnn_input = embed_x[mask]
                domain_label = targets[mask]
            domain_dnn_input = domain_bn(domain_dnn_input, self.shared_bn_weight, self.shared_bn_bias)


            for i, (domain_linear_i, shared_linear_i) in enumerate(zip(domain_dnn.linears, self.shared_dnn.linears)):
                weight_i = domain_linear_i.weight * shared_linear_i.weight
                bias_i = domain_linear_i.bias + shared_linear_i.bias
                out = F.linear(domain_dnn_input, weight_i, bias_i)
                if out.shape[0] > 1:
                    out = domain_dnn.bn[i](out)
                out = domain_dnn.activation_layers[i](out)
                out = domain_dnn.dropout(out)
                domain_dnn_input = out

            weight_linear = domain_dnn_linear.weight * self.shared_dnn_linear.weight
            bias_linear = domain_dnn_linear.bias + self.shared_dnn_linear.bias
            domain_dnn_logit = F.linear(domain_dnn_input, weight_linear, bias_linear)
            for other in other_outs:
                if x_group is None:
                    domain_dnn_logit += other
                else:
                    domain_dnn_logit += other[mask]
            ys.append(self.output_layers[group_id](domain_dnn_logit))
            grouped_targets.append(domain_label)

        if x_group is None:
            return torch.cat(ys, dim=1)
        else:
            return torch.cat(ys, dim=0), torch.cat(grouped_targets, dim=0)


class MDR_BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MDR_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: Tensor, shared_weight, shared_bias) -> Tensor:
        if input.shape[0] == 1:
            return input  # This is a hack to avoid batchnorm error when batch size is 1

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * shared_weight,
            self.bias + shared_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )