#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base Model for all models.
    Initiate embedding layer, linear layer, regularization, etc.
    Reference: https://github.com/shenweichen/DeepCTR-Torch
    """
    def __init__(self, feature_dims, embed_dim, l2_reg_embedding=1e-5, l2_reg_linear=1e-5):
        super(BaseModel, self).__init__()
        self.feature_dims = feature_dims
        self.embedding = FeaturesEmbedding(feature_dims, embed_dim)
        self.embed_output_dim = self.embedding.output_dim0 * embed_dim
        self.embed_dim = embed_dim
        self.field_num = self.embedding.field_num

        self.linear = FeaturesLinear(self.embed_output_dim)  # 线性+偏置

        self.is_concat_linear_cn = None

        self.reg_loss = torch.zeros((1,))
        self.regularization_weight = []

        self.add_regularization_weight(self.embedding.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.linear.named_parameters()), l2=l2_reg_linear)

    def build_tower_output(self, n_tower, tower_input_dim, tower_dims, dropout):
        """
        build towers, towers_linear, output_layers for multi-tower model with dcn; add tower reg weight
        """
        # the first layer expert dim is the input data dim other expert dim
        towers = nn.ModuleList(
            MultiLayerPerceptron(tower_input_dim, tower_dims, dropout, output_layer=True)
            for _ in range(self.n_tower))
        output_layers = nn.ModuleList([nn.Sigmoid() for _ in range(n_tower)])
        towers_linear = None

        return towers, towers_linear, output_layers

    def tower_forward(self, tower_inputs, other_outs=None):
        ys = []
        for tower_input, tower, predict_layer in zip(tower_inputs, self.towers, self.output_layers):
            y_logits = tower(tower_input)
            if other_outs is not None:
                for other in other_outs:
                    y_logits += other
            ys.append(predict_layer(y_logits))
        return torch.cat(ys, dim=1)

    def build_atten(self, config, dropout):
        atten_embed_dim = getattr(config, 'atten_embed_dim', self.embed_dim)
        self.atten_embedding = torch.nn.Linear(self.embed_dim, atten_embed_dim)
        self.atten_output_dim = self.embedding.output_dim0 * atten_embed_dim
        self.att_res = config.att_res
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(config.atten_embed_dim, config.att_head_num, dropout=dropout) for _ in
            range(config.att_layer_num)
        ])
        if self.att_res:
            self.V_res_embedding = torch.nn.Linear(self.embed_dim, atten_embed_dim)
        self.atten_linear = nn.Linear(self.atten_output_dim, 1, bias=False)

    def atten_forward(self, embed_x):
        embed_x = embed_x.reshape(-1, self.field_num, self.embed_dim)
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
        return self.atten_linear(cross_term)

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, device):
        total_reg_loss = torch.zeros((1,), device=device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss


class FeaturesLinear(torch.nn.Module):
    # 线性项+偏置项
    def __init__(self, field_dims, output_dim=1, sigmoid=False):
        super().__init__()
        self.fc = nn.Linear(field_dims, output_dim, bias=True)
        self.sigmoid = sigmoid

    def forward(self, x):
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()

        self.field_num = len(field_dims)
        self.output_dim0 = self.field_num
        self.embed_dim = embed_dim
        print(f'field_num: {self.field_num}, ', end='')
        print(f'embed_dim: {self.embed_dim}')

        self.embedding_dict = torch.nn.Embedding(sum(field_dims), embed_dim)  # Embedding 也是继承自module，有可学习参数
        if np.__version__ >= '1.20.0':
            self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        else:
            self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)  # 累加除了itemID的各个field长度
        # torch.nn.init.xavier_uniform_(self.embedding_dict.weight.data)

    def forward(self, x, squeeze_dim=False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return: embedding tensor of size ``(batch_size, output_dim0, embed_dim)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)   # 索引的偏移量
        embed_x = self.embedding_dict(x)
        if squeeze_dim:
            embed_x = torch.flatten(embed_x, start_dim=1)

        return embed_x


class FactorizationMachine(torch.nn.Module):
    # 求的是二阶特征交互的项
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:  # 在embed_dim方向上求和
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, layer_dims, dropout, output_layer=True, bn=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for layer_dim in layer_dims:
            self.layers.append(torch.nn.Linear(input_dim, layer_dim))
            if bn:
                self.layers.append(torch.nn.BatchNorm1d(layer_dim))
            self.layers.append(torch.nn.ReLU())  # 使用 ReLU 激活函数
            self.layers.append(torch.nn.Dropout(p=dropout))
            input_dim = layer_dim

        if output_layer:
            self.layers.append(torch.nn.Linear(input_dim, 1))

        # self.mlp = torch.nn.Sequential(*layers)  # list带星号，解开成独立的参数，传入函数；dict同理，key需要与形参名字相同

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        for layer in self.layers:
            if isinstance(layer, torch.nn.BatchNorm1d) and x.shape[0] == 1:
                # 当 batch size 为 1 时跳过 BatchNorm 层
                continue
            x = layer(x)
        return x


def activation_layer(act_name):
    """
    Ref: https://github.com/shenweichen/DeepCTR-Torch
    Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = nn.Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class DNN(nn.Module):
    """
      Ref: https://github.com/shenweichen/DeepCTR-Torch
      The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', dropout_rate=0, use_bn=True):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for _ in range(len(hidden_units) - 1)])


    def forward(self, inputs):
        deep_input = inputs
        batch_size = inputs.shape[0]

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn and batch_size > 1:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class CrossNetwork(nn.Module):
    """CrossNetwork  mentioned in the DCN paper.

    Args:
        input_dim (int): input dim of input tensor

    Shape:
        - Input: `(batch_size, *)`
        - Output: `(batch_size, *)`

    """

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            x = x0 * self.w[i](x) + self.b[i] + x
        return x


class CrossNetMix(nn.Module):
    """ CrossNetMix improves CrossNetwork by:
        1. add MOE to learn feature interactions in different subspaces
        2. add nonlinear transformations in low-dimensional space
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
    """

    def __init__(self, input_dim, num_layers=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        # U: (input_dim, low_rank)
        self.u_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, input_dim, low_rank))) for i in range(self.num_layers)])
        # V: (input_dim, low_rank)
        self.v_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, input_dim, low_rank))) for i in range(self.num_layers)])
        # C: (low_rank, low_rank)
        self.c_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, low_rank, low_rank))) for i in range(self.num_layers)])
        self.gating = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for i in range(self.num_experts)])

        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(
            torch.empty(input_dim, 1))) for i in range(self.num_layers)])

    def forward(self, x):
        x_0 = x.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.num_layers):
            output_of_experts = []
            gating_score_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.v_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.c_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.u_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_experts = torch.stack(gating_score_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l


class Expert(torch.nn.Module):
    def __init__(self, input_dim, layer_dims, dropout):
        super(Expert, self).__init__()
        layers = list()
        for layer_dim in layer_dims[:-1]:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.BatchNorm1d(layer_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = layer_dim
        layers.append(torch.nn.Linear(input_dim, layer_dims[-1]))
        self.expert = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim * sparse_feature_num + dense_feature_num)``
        """
        return self.expert(x)


class Tower(torch.nn.Module):
    def __init__(self, input_dim, layer_dims, dropout):
        super(Expert, self).__init__()
        layers = list()
        for layer_dim in layer_dims[:-1]:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = layer_dim
        layers.append(torch.nn.Linear(input_dim, layer_dims[-1]))
        self.tower = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.tower(x)


class InnerProductNetwork(torch.nn.Module):

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)  # 结果应该跟FactorizationMachine一样


class OuterProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':  # opnn里论文是outer product，这里实际改成了kernel product
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]  # size (batch_size, num_ix, embed_dim)
        if self.kernel_type == 'mat':  # (batch_size, 1, num_ix, embed_dim) * (embed_dim, num_ix, embed_dim) = (batch_size, embed_dim, num_ix, embed_dim)
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)  # permute, 将tensor的维度换位，相当于转置
            # kp.shape = (batch_size, num_ix, embed_dim)
            return torch.sum(kp * q, -1)  # 二阶交互后的矩阵求和后作为标量返回
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):  # init里是网络里会更新的参数
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)  # 不用ParameterList大概是直接乘不方便
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)  # x^T * w 其实就是各加权后求和
            x = x0 * xw + self.b[i] + x
        return x


class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)  # 即2层网络，embed_dim - attn_size - 1
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]  # p,q是二阶特征交互的组合
        inner_product = p * q        # Pair-wise Interaction Layer
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)  # 对所有权重归一化
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)  # 用functional.dropout的时候一定要设training等于模型的training
        attn_output = torch.sum(attn_scores * inner_product, dim=1)   # attention-based pooling
        attn_output = F.dropout(attn_output, p=self.dropouts[1], training=self.training)
        return self.fc(attn_output)


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2  # 对半压缩
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc_input_dim = fc_input_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)  # 通过unsequuze实现外积
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)  # embed_dim始终没变，因此是vector-wise
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        # return self.fc(torch.sum(torch.cat(xs, dim=1), 2))
        return torch.sum(torch.cat(xs, dim=1), 2)


class AnovaKernel(torch.nn.Module):

    def __init__(self, order, reduce_sum=True):
        super().__init__()
        self.order = order
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        batch_size, num_fields, embed_dim = x.shape
        a_prev = torch.ones((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)  # DP table的第一行
        for t in range(self.order):
            a = torch.zeros((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
            a[:, t+1:, :] += x[:, t:, :] * a_prev[:, t:-1, :]  # t+1是因为只更新j>t的部分，这里=应该也可以
            a = torch.cumsum(a, dim=1)  # 对应原文的累加
            a_prev = a
        if self.reduce_sum:
            return torch.sum(a[:, -1, :], dim=-1, keepdim=True)
        else:
            return a[:, -1, :]