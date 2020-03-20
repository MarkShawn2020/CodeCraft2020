# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/19 23:54
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
import os
import logging
import pickle
import numpy as np
from .functions import sigmoid, cross_entropy, mse


class Model:

	def __init__(self, lr=0.03):
		self.lr = lr

	def fit(self, X, Y):
		raise NotImplementedError("该函数必须继承实现！")

	def predict(self, X):
		raise NotImplementedError("该函数必须继承实现！")

	@staticmethod
	def evaluate(Y_pred, Y_target):
		raise NotImplementedError("该函数必须继承实现！")

	def evaluate_data_loader(self, data_loader):
		return self.evaluate(self.predict(data_loader.X), data_loader.Y)

	def save_prediction(self, Y_pred, path, delimiter="\n"):
		with open(path, 'w') as f:
			for each_Y_pred in Y_pred:
				f.write("{}{}".format(each_Y_pred, delimiter))
		logging.info("Saved prediction to file {}".format(os.path.abspath(path)))


class LogisticRegression(Model):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def fit(self, X, Y_target):
		Y_pred = sigmoid(X @ self.w)
		self.w += self.lr * (Y_target - Y_pred).T @ X / len(X)

	def predict(self, X):
		Y_pred = ((X @ self.w) > 0).astype(int)
		return Y_pred

	@staticmethod
	def evaluate(Y_pred, Y_target) -> dict:
		loss_cross_entropy = cross_entropy(Y_pred, Y_target)
		n_errs = (Y_pred.astype(int) ^ Y_target.astype(int)).sum().item()
		N = len(Y_target)
		acc = (1 - n_errs / N) * 100
		return {
			"n_errs": n_errs,
			"N_items": N,
			"acc": acc,
			"loss": loss_cross_entropy,
		}

	def init_weight(self, n_features, scale=0.01):
		self.w = np.random.random((n_features, )) * scale

	def dump_weight(self, path):
		pickle.dump(self.w, open(path, "wb"))
		logging.info("Dumped weights into file {}".format(os.path.abspath(path)))

	def load_weight(self, path):
		self.w = pickle.load(open(path, "rb"))
		logging.info("Loaded weights from file {}".format(os.path.abspath(path)))

