# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/19 23:07
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
import os
import logging
import random
import numpy as np
from functools import partial

from .common import lazy


class DataLoader:
	def __init__(self, use_mp=True, shuffle=False, batch_size=256, split_ratio=0.9):
		self.use_mp = use_mp
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.split_ratio = split_ratio
		assert 0 <= split_ratio <= 1, "训练集与验证集之间的切割比率要在0-1之间！"

	def _load_from_file(self, file_path, dtype=float):
		assert os.path.exists(file_path), "目标文件不存在: {}".format(os.path.abspath(file_path))
		with open(file_path, "r") as fp:
			lines = fp.readlines()
			if self.use_mp:
				import multiprocessing as mp
				with mp.Pool() as p:
					data = np.array(p.map(partial(self._load_line, dtype=dtype), lines))
			else:
				data = np.array(list(map(partial(self._load_line, dtype=dtype), lines)))
		logging.info("Loaded data with shape {} from {}".format(data.shape, os.path.abspath(file_path)))
		return data

	def load_X(self, file_path):
		self.X = self._load_from_file(file_path, dtype=float)
		self.N_items, self.N_features = self.X.shape

	def load_Y(self, file_path):
		self.Y = self._load_from_file(file_path, dtype=int).flatten()

	def load_XY(self, file_path):
		data = self._load_from_file(file_path, dtype=float)
		self.X = data[:, :-1]
		self.N_items, self.N_features = self.X.shape
		self.Y = data[:, -1].astype(int)

	@lazy
	def data(self):
		_data = np.hstack([self.X, self.Y.reshape(-1, 1)])
		return _data

	@lazy
	def X_to_valid(self):
		return self.X[int(self.N_items * self.split_ratio):]

	@lazy
	def Y_to_valid(self):
		return self.Y[int(self.N_items * self.split_ratio):]

	@lazy
	def N_to_train(self):
		return int(self.N_items * self.split_ratio)

	@lazy
	def _train_slice(self) -> list:
		"""
		使用数组的索引以操控shuffle
		预期可以比直接shuffle训练数据效率更高

		:return: 返回一个索引列表，该列表不包含验证集部分
		"""
		idx = list(range(self.N_items))
		if self.shuffle:
			random.shuffle(idx)
		return idx[: self.N_to_train]

	@staticmethod
	def _load_line(line, delimiter=",", dtype=float):
		return np.array(line.split(delimiter), dtype=dtype)

	def __iter__(self):
		for i in range(0, self.N_to_train, self.batch_size):
			yield self.X[self._train_slice[i: i + self.batch_size]], \
			      self.Y[self._train_slice[i: i + self.batch_size]]

	def __len__(self):
		import math
		N_batches =  math.ceil(self.N_to_train / self.batch_size)
		del math
		return N_batches
