# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/20 20:22
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------

import sys, os
sys.path.insert(0, os.path.join(__file__, "../.."))

from core.dataloaders import DataLoader
from core.models import LogisticRegression
from core.common import *


@calc_time
def train():
	for epoch in range(EPOCHS):
		for X, Y in train_data_loader:
			my_model.fit(X, Y)

			Y_to_valid_pred = my_model.predict(train_data_loader.X_to_valid)
			result = my_model.evaluate(Y_to_valid_pred, train_data_loader.Y_to_valid)
			print("Epoch: {}, Result: {}".format(epoch, result))


if __name__ == '__main__':
	"""
	每个函数按 Ctrl + P 可以查看它的默认参数
	"""
	BATCH_SZIE = 100
	LR = 0.05
	EPOCHS = 10

	train_data_loader = DataLoader(batch_size=BATCH_SZIE,
	                               select_ratio=0.2, split_ratio=0.9, shuffle=True, standardize_x=True)
	train_data_loader.load_XY(train_data_path)

	my_model = LogisticRegression(lr=LR)
	my_model.init_weight(train_data_loader.N_features, scale=0.01)

	train()

	test_data_loader = DataLoader()
	test_data_loader.load_X(test_data_path)
	test_data_loader.load_Y(test_answer_path)

	result = my_model.evaluate_data_loader(test_data_loader)
	print(result)



