#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_auc_score
import numpy as np
from collections import defaultdict
import tqdm


def get_user_pred(y_true, y_pred, users):
    """
        divide the result into different group by user id
        Ref: https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/basic/metric.py
        Args:
            y_true (array): all true labels of the data
            y_pred (array): the predicted score
            users (array): user id

        Return:
            user_pred (dict): {userid: values}, key is user id and value is the labels and scores of each user
    """
    user_pred = {}
    for i, u in enumerate(users):
        if u not in user_pred:
            user_pred[u] = {'y_true': [y_true[i]], 'y_pred': [y_pred[i]]}
        else:
            user_pred[u]['y_true'].append(y_true[i])
            user_pred[u]['y_pred'].append(y_pred[i])

    return user_pred


def gauc_score(y_true, y_pred, users, weights=None):
    """
        compute GAUC
        Ref: https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/basic/metric.py

        Args:
            y_true (array): dim(N, ), all true labels of the data
            y_pred (array): dim(N, ), the predicted score
            users (array): dim(N, ), user id
            weight (dict): {userid: weight_value}, it contains weights for each group.
                    if it is None, the weight is equal to the number
                    of times the user is recommended
        Return:
            score: float, GAUC
    """
    assert len(y_true) == len(y_pred) and len(y_true) == len(users)

    user_pred = get_user_pred(y_true, y_pred, users)
    score = 0
    num = 0
    for u in tqdm.tqdm(user_pred.keys(), desc='Computing GAUC with each users', mininterval=30):
        if not ((0 in user_pred[u]['y_true']) and (1 in user_pred[u]['y_true'])):
            continue  # No positive or negative samples in some users, these users have been removed from GAUC
        auc = roc_auc_score(user_pred[u]['y_true'], user_pred[u]['y_pred'])
        if weights is None:
            user_weight = len(user_pred[u]['y_true'])
        else:
            user_weight = weights[u]
        auc *= user_weight
        num += user_weight
        score += auc
    return score / num
