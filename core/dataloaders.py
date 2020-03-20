# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/19 23:07
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
import os
import mmap
import logging
import random
import numpy as np
from functools import partial

from .common import lazy


class DataLoader:
	def __init__(self, use_mp=True, use_mmap=True, shuffle=False, standardize_x=True, batch_size=256,
	             select_ratio=1, split_ratio=0.9, seed=None):
		self.use_mp = use_mp
		self.use_mmap = use_mmap
		self.shuffle = shuffle
		self.standardize_x = standardize_x
		self.batch_size = batch_size

		self.select_ratio = select_ratio
		assert 0 <= select_ratio <= 1, "训练集的选取比率要在0-1之间！"

		self.split_ratio = split_ratio
		assert 0 <= split_ratio <= 1, "训练集与验证集之间的切割比率要在0-1之间！"

		self.seed = seed
		np.random.seed(self.seed)

	def _load_from_file(self, file_path, dtype=float):
		assert os.path.exists(file_path), "目标文件不存在: {}".format(os.path.abspath(file_path))
		with open(file_path, "r") as fp:
			if self.use_mmap:
				m = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
				# 注意，由于mmap没有readlines方法，所以最后一行空行要去掉
				all_lines = m.read().split(b'\n')[:-1]
				func_load_line = self._load_line_bytes
			else:
				all_lines = fp.readlines()
				func_load_line = self._load_line

			N_lines_all = len(all_lines)
			N_lines_selected = int(N_lines_all * self.select_ratio)
			logging.info("Loaded lines [{}/{}] with SELECT_RATIO: {}".format(
				N_lines_selected, N_lines_all, self.select_ratio))

			if self.shuffle:
				lines = random.sample(all_lines, N_lines_selected)
			else:
				lines = all_lines[: N_lines_selected]

			if self.use_mp:
				import multiprocessing as mp
				with mp.Pool() as p:
					data = np.array(p.map(partial(func_load_line, dtype=dtype), lines))
			else:
				data = np.array(list(map(partial(func_load_line, dtype=dtype), lines)))
		logging.info("Loaded data with shape {} from {}".format(data.shape, os.path.abspath(file_path)))
		return data

	def load_X(self, file_path):
		self.X = self._load_from_file(file_path, dtype=float)
		if self.standardize_x:
			self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
		self.N_items, self.N_features = self.X.shape

	def load_Y(self, file_path):
		self.Y = self._load_from_file(file_path, dtype=int).flatten()

	def load_XY(self, file_path):
		data = self._load_from_file(file_path, dtype=float)
		self.X = data[:, :-1]
		if self.standardize_x:
			self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
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

	@staticmethod
	def _load_line(line, delimiter=",", dtype=float):
		return np.array(line.split(delimiter), dtype=dtype)

	@staticmethod
	def _load_line_bytes(line, delimiter=b",", dtype=float):
		return np.array(line.split(delimiter), dtype=dtype)

	def __iter__(self):
		for i in range(0, self.N_to_train, self.batch_size):
			yield self.X[i: i + self.batch_size], \
			      self.Y[i: i + self.batch_size]

	def __len__(self):
		return np.ceil(self.N_to_train / self.batch_size).astype(int).item()
