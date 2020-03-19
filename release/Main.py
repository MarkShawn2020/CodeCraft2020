






class lazy:
	def __init__(self, func):
		self.func = func

	def __get__(self, instance, cls):
		val = self.func(instance)
		setattr(instance, self.func.__name__, val)
		return val

import os, sys
if "win" in sys.platform:
	data_dir = os.path.join(__file__, "../../data")
	train_data_path = os.path.join(data_dir, "train_data.txt")
	test_data_path = os.path.join(data_dir, "test_data.txt")
	test_answer_path = os.path.join(data_dir, "answer.txt")
	test_predict_path = os.path.join(data_dir, "result.txt")
else:
	train_data_path = "/data/train_data.txt"
	test_data_path = "/data/test_data.txt"
	test_answer_path = "/data/answer_data.txt"
	test_predict_path = "/projects/student/result.txt"
	
	
def calc_time(func):
	import time
	def wrapper(*args, **kwargs):
		st = time.time()
		result = func(*args, **kwargs)
		dt = time.time() - st
		print("Func: {}, Time: {}".format(func.__name__, dt))
		return result if result else dt
	return wrapper






import numpy as np

EPS = 1e-8


def sigmoid(X):
	return 1 / (1 + np.exp(-X))


def mse(Y_pred, Y_target):
	return ((Y_pred - Y_target) ** 2).mean()


def cross_entropy(Y_pred, Y_target):
	return -(Y_target * np.log(Y_pred+EPS) + (1-Y_target) * np.log(1-Y_pred+EPS)).mean()





import os
import logging
import random
import numpy as np
from functools import partial




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






import os
import logging
import pickle
import numpy as np



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

	def init_weight(self, n_features):
		self.w = np.zeros((n_features, ))

	def dump_weight(self, path):
		pickle.dump(self.w, open(path, "wb"))
		logging.info("Dumped weights into file {}".format(os.path.abspath(path)))

	def load_weight(self, path):
		self.w = pickle.load(open(path, "rb"))
		logging.info("Loaded weights from file {}".format(os.path.abspath(path)))











import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(process)d %(name)s: %(message)s")


@calc_time
def train(model, train_data_loader):
	
	logging.info("Start Training!")
	max_iterations = MAX_ITERATIONS
	for epoch in range(EPOCHS):
		for i, (X, Y) in enumerate(train_data_loader, 0):
			model.fit(X, Y)

			if LOG_LEVEL <= logging.DEBUG and (i + 1) % LOG_INTERVAL == 0:
				eval_result = model.evaluate(model.predict(train_data_loader.X_to_valid), train_data_loader.Y_to_valid)
				logging.debug("epoch: [{}/{}], iter: [{}/{}], err: [{}/{}], acc: {:.2f}%, loss: {:.6f}".format(
					(epoch + 1), EPOCHS, i + 1, len(train_data_loader), *eval_result.values()))
			max_iterations -= 1
			if max_iterations < 0:
				logging.info("Stopped training for reaching max iterations of {}".format(MAX_ITERATIONS))
				return
	else:
		logging.info("Stopped training for reaching max epochs of {}".format(EPOCHS))
		return


@calc_time
def main():
	
	train_data_loader = DataLoader(
		shuffle=SHUFFLE, use_mp=ENABLE_MULTI_PROCESSES,
		batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO)
	train_data_loader.load_XY(train_data_path)

	
	lr = LogisticRegression(lr=LR)
	lr.init_weight(train_data_loader.N_features)

	
	train(lr, train_data_loader)

	
	test_data_loader = DataLoader(use_mp=ENABLE_MULTI_PROCESSES)
	test_data_loader.load_X(test_data_path)
	test_data_loader.load_Y(test_answer_path)

	
	Y_pred = lr.predict(test_data_loader.X)
	lr.save_prediction(Y_pred, path=test_predict_path)

	
	if LOG_LEVEL <= logging.INFO:
		test_result = lr.evaluate(Y_pred, test_data_loader.Y)
		logging.info("[TEST RESULT] err: [{}/{}], acc: {:.2f}%".format(*test_result.values()))
		lr.dump_weight(WEIGHTS_PATH)


if __name__ == '__main__':
	ENABLE_MULTI_PROCESSES = True
	SHUFFLE = True  
	SPLIT_RATIO = 0.9  
	LOG_INTERVAL = 10
	WEIGHTS_PATH = "w.pkl"
	MAX_ITERATIONS = 100000  

	"""
	经测试比较好的结果是
	"""
	LR = 0.01
	BATCH_SIZE = 10
	EPOCHS = 10

	LOG_LEVEL = logging.WARNING
	main()

