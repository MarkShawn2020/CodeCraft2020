# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/20 0:15
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
import numpy as np

EPS = 1e-8


def sigmoid(X):
	return 1 / (1 + np.exp(-X))


def mse(Y_pred, Y_target):
	return ((Y_pred - Y_target) ** 2).mean()


def cross_entropy(Y_pred, Y_target):
	return -(Y_target * np.log(Y_pred+EPS) + (1-Y_target) * np.log(1-Y_pred+EPS)).mean()