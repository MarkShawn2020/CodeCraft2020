# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/20 0:49
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
from R0_Plus.core import dataloader, models
from R0_Plus.core.common import *

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(process)d %(name)s: %(message)s")


import pickle
import numpy as np


@calc_time
def train(model, train_data_loader, EPOCHS):
	# 模型训练
	logging.info("Start Training!")
	max_iterations = MAX_ITERATIONS / train_data_loader.batch_size
	iter = 0
	for epoch in range(EPOCHS):
		for i, (X, Y) in enumerate(train_data_loader, 0):
			model.fit(X, Y)

			if LOG_LEVEL <= logging.DEBUG and (i+1) % LOG_INTERVAL == 0:
				eval_result = model.evaluate(model.predict(train_data_loader.X_to_valid), train_data_loader.Y_to_valid)
				logging.debug("epoch: [{}/{}], iter: [{}/{}], err: [{}/{}], acc: {:.2f}%, loss: {:.6f}".format(
					(epoch+1), EPOCHS, i+1, len(train_data_loader),*eval_result.values()))
			iter += 1
			if iter > max_iterations:
				logging.info("Stopped training for reaching max iterations of {}".format(max_iterations))
				return
	else:
		logging.info("Stopped training for reaching max epochs of {}".format(EPOCHS))
		return

@calc_time
def grid_test():
	result = []
	try:
		for lr in np.logspace(-4, 0, 5):
			model = models.LogisticRegression(lr=lr)
			for batch_size in np.logspace(0, 3, 4, dtype=int):
				train_data_loader.batch_size = batch_size
				for epochs in np.logspace(0, 4, 5, dtype=int):
					model.init_weight(train_data_loader.N_features)

					period = {
						"LR": lr,
						"BATCH_SIZE": batch_size,
						"EPOCHS": epochs,
					}
					print("\n", period)
					train_time = train(model, train_data_loader, epochs)

					eval_data = model.evaluate_data_loader(test_data_loader)
					print(eval_data)
					eval_data.update(period)
					logging.info(eval_data)
					eval_data.update({"TRAIN_TIME": train_time})
					result.append(eval_data)
	finally:
		return result


if __name__ == '__main__':
	LOG_LEVEL = logging.INFO
	ENABLE_MULTI_PROCESSES = True
	SHUFFLE     = True      # 是否打乱训练数据顺序
	EPOCHS = 1              # 训练集还是要至少跑一整遍的，不然实在太流氓了
	SPLIT_RATIO = 0.9       # 切割训练集与验证集比率
	MAX_ITERATIONS = 1000000 # 预期迭代次数计算公式： N_to_train / BS * Epochs
	LOG_INTERVAL = 10
	WEIGHTS_PATH = "w.pkl"

	# 加载训练集
	train_data_loader = dataloader.DataLoader(
		shuffle=SHUFFLE, use_mp=ENABLE_MULTI_PROCESSES,
		batch_size=1, split_ratio=SPLIT_RATIO)
	train_data_loader.load_XY(train_data_path)

	# 加载预测集
	test_data_loader = dataloader.DataLoader(use_mp=ENABLE_MULTI_PROCESSES)
	test_data_loader.load_X(test_data_path)
	test_data_loader.load_Y(test_answer_path)

	np.random.seed(1)
	result = grid_test()
	pickle.dump(result, open("eval_data.pkl", "wb"))


