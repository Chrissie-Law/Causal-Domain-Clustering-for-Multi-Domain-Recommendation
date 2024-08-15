#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor
from model.layer import BaseModel, CrossNetwork, DNN, MultiLayerPerceptron
from torch.nn.modules.batchnorm import _NormBase
import torch.nn.functional as F


class ADL(BaseModel):
    """
    Reference:
        Li, Jinyun, et al.
        "ADL: Adaptive Distribution Learning Framework for Multi-Scenario CTR Prediction."  SIGIR 2023.
    """
    def __init__(self, feature_dims, embed_dim,
                 n_tower, tower_dims, domain_idx=None, dropout=0.2,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5,
                 dlm_iters=3, dlm_update_rate=0.9, device=None, config=None):
        super(ADL, self).__init__(feature_dims, embed_dim,
                                  l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = 'adl'
        self.n_tower = n_tower
        self.domain_idx = domain_idx
        self.device = device
        self.cluster_num = n_tower
        self.dlm_iters = dlm_iters
        self.dlm_update_rate = dlm_update_rate
        self.cluster_centers = torch.randn((self.cluster_num, self.embed_output_dim)).to(self.device)
        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)
        print("ADL cluster_num:", self.cluster_num, "dlm_iters:", self.dlm_iters,
              "dlm_update_rate:", self.dlm_update_rate)

        if self.use_dcn:
            self.cn = CrossNetwork(self.embed_output_dim, config.n_cross_layers)
        if self.use_atten:
            self.build_atten(config, dropout)

        self.domain_mlps = nn.ModuleList([
            # DNN(self.embed_output_dim, tower_dims, dropout_rate=dropout)
            MultiLayerPerceptron(self.embed_output_dim, tower_dims, dropout, output_layer=False)
            for _ in range(n_tower)])

        self.domain_mlps_linears = nn.ModuleList([nn.Linear(tower_dims[-1], 1)
                                                  for _ in range(n_tower)])

        self.shared_mlps = MultiLayerPerceptron(self.embed_output_dim, tower_dims, dropout, output_layer=False)
        self.shared_mlps_linear = nn.Linear(tower_dims[-1], 1)

        self.output_layers = nn.ModuleList([nn.Sigmoid() for _ in range(n_tower)])

        if self.use_dcn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.domain_mlps.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.shared_mlps.named_parameters()), l2=l2_reg_dnn)

    def DLM_routing(self, embed_x):
        """
        implement DLM routing by updating cluster centers
        @param embed_x: [batch_size, embed_dim]
        @return: distribution_coefficients [batch_size, cluster_num]
        """
        with torch.no_grad():
            for t in range(self.dlm_iters):
                similarity_scores = torch.matmul(embed_x, self.cluster_centers.t())
                distribution_coefficients = F.softmax(similarity_scores, dim=1)  # [batch_size, cluster_num]

                weighted_sum = torch.matmul(distribution_coefficients.t(), embed_x)  # [cluster_num, embed_dim]
                tmp_cluster_centers = F.normalize(weighted_sum, p=2, dim=1)

            self.cluster_centers = F.normalize(
                self.dlm_update_rate * self.cluster_centers + (1 - self.dlm_update_rate) * tmp_cluster_centers, p=2, dim=1)
        return distribution_coefficients

    def forward(self, x, group=None, targets=None, is_training=True):
        embed_x = self.embedding(x, squeeze_dim=True)

        distribution_coefficients = self.DLM_routing(embed_x)

        x_to_tower = torch.argmax(distribution_coefficients, dim=1)

        ys = []
        grouped_targets = []
        if not is_training:
            ys_tensor = torch.zeros((embed_x.size(0), 1), dtype=torch.float, device=self.device)

        # shared_mlp_input = embed_x
        # shared_mlp_out = self.shared_mlps(shared_mlp_input)

        other_outs = [self.linear(embed_x)]
        if self.use_dcn:
            cn_out = self.cn(embed_x)
            other_outs.append(cn_out)
        if self.use_atten:
            atten_out = self.atten_forward(embed_x)
            other_outs.append(atten_out)

        for tower_id in range(self.n_tower):
            domain_mlp = self.domain_mlps[tower_id]
            domain_mlp_linears = self.domain_mlps_linears[tower_id]

            mask = (x_to_tower == tower_id)  # TODO: x_group
            domain_mlp_input = embed_x[mask]

            domain_dnn_out = domain_mlp(domain_mlp_input)
            weight_linear = domain_mlp_linears.weight * self.shared_mlps_linear.weight
            bias_linear = domain_mlp_linears.bias + self.shared_mlps_linear.bias

            domain_dnn_logit = F.linear(domain_dnn_out, weight_linear, bias_linear)
            for other in other_outs:
                domain_dnn_logit += other[mask]

            if is_training:
                ys.append(self.output_layers[tower_id](domain_dnn_logit))
                grouped_targets.append(targets[mask])
            else:
                ys_tensor[mask] = self.output_layers[tower_id](domain_dnn_logit)

        if is_training:
            return torch.cat(ys, dim=0), torch.cat(grouped_targets, dim=0)
        else:
            return ys_tensor


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
