# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/20 20:22
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------


from MachineLarning_Numpy_CodeCraft2020.core.common import *
from MachineLarning_Numpy_CodeCraft2020.core.dataloaders import DataLoader
from MachineLarning_Numpy_CodeCraft2020.core.models import LogisticRegression


if __name__ == '__main__':
	"""
	每个函数按 Ctrl + P 可以查看它的默认参数
	"""

	train_data_loader = DataLoader(batch_size=1000, )
	train_data_loader.load_XY(train_data_path)

	class MyModel(LogisticRegression):
		"""
		可以定义自己的fit函数
		"""
		pass

	my_model = MyModel(lr=0.05)
	my_model.init_weight(train_data_loader.N_features)

	for epoch in range(10):
		for X, Y in train_data_loader:
			my_model.fit(X, Y)

			Y_to_valid_pred = my_model.predict(train_data_loader.X_to_valid)
			result = my_model.evaluate(Y_to_valid_pred, train_data_loader.Y_to_valid)
			print("Epoch: {}, Result: {}".format(epoch, result))

	test_data_loader = DataLoader()
	test_data_loader.load_X(test_data_path)
	test_data_loader.load_Y(test_answer_path)

	result = my_model.evaluate_data_loader(test_data_loader)
	print(result)



