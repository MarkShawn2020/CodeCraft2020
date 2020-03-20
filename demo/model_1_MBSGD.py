# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/20 0:49
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
from MachineLarning_Numpy_CodeCraft2020.core.dataloaders import DataLoader
from MachineLarning_Numpy_CodeCraft2020.core.models import LogisticRegression
from MachineLarning_Numpy_CodeCraft2020.core.common import *

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(process)d %(name)s: %(message)s")


@calc_time
def train(model, train_data_loader):
	# 模型训练
	logging.info("Start Training!")
	max_iterations = MAX_ITERATIONS
	for epoch in range(EPOCHS):
		for i, (X, Y) in enumerate(train_data_loader, 0):
			model.fit(X, Y)

			if LOG_LEVEL <= logging.DEBUG and (i+1) % LOG_INTERVAL == 0:
				eval_result = model.evaluate(model.predict(train_data_loader.X_to_valid), train_data_loader.Y_to_valid)
				logging.debug("epoch: [{}/{}], iter: [{}/{}], err: [{}/{}], acc: {:.2f}%, loss: {:.6f}".format(
					(epoch+1), EPOCHS, i+1, len(train_data_loader),*eval_result.values()))
			max_iterations -= 1
			if max_iterations < 0:
				logging.info("Stopped training for reaching max iterations of {}".format(MAX_ITERATIONS))
				return
	else:
		logging.info("Stopped training for reaching max epochs of {}".format(EPOCHS))
		return


@calc_time
def load_train_data():
	train_data_loader = DataLoader(
		shuffle=SHUFFLE, use_mp=ENABLE_MULTI_PROCESSES, batch_size=BATCH_SIZE,
		select_ratio=SELECT_RATIO, split_ratio=SPLIT_RATIO)
	train_data_loader.load_XY(train_data_path)
	return train_data_loader


@calc_time
def main():
	# 加载训练集
	train_data_loader = load_train_data()

	# 模型初始化
	lr = LogisticRegression(lr=LR)
	lr.init_weight(train_data_loader.N_features)

	# 模型训练
	train(lr, train_data_loader)

	# 加载预测集
	test_data_loader = DataLoader(use_mp=ENABLE_MULTI_PROCESSES)
	test_data_loader.load_X(test_data_path)

	# 模型预测
	Y_pred = lr.predict(test_data_loader.X)
	lr.save_prediction(Y_pred, path=test_predict_path)

	# 模型评估与持久化
	if LOG_LEVEL <= logging.INFO:
		test_data_loader.load_Y(test_answer_path)
		test_result = lr.evaluate_data_loader(test_data_loader)
		logging.info("[TEST RESULT] err: [{}/{}], acc: {:.2f}%".format(*test_result.values()))
		lr.dump_weight(WEIGHTS_PATH)



if __name__ == '__main__':

	# 根据平台控制程序的日志级别，设置成WARNIGN基本可以避免很多输出开销
	LOG_LEVEL = logging.DEBUG if 'win' in sys.platform else logging.WARNING

	# 是否启用多进程加载文件，在鲲鹏64核的帮助下此有奇效
	ENABLE_MULTI_PROCESSES = True

	SHUFFLE     = True      # 是否打乱训练数据顺序

	WEIGHTS_PATH = os.path.join(DATA_DIR, "w.pkl")

	SELECT_RATIO = 0.2
	SPLIT_RATIO = 0.9       # 切割训练集与验证集比率
	LOG_INTERVAL = 10
	MAX_ITERATIONS = 100000 # 预期迭代次数计算公式： N_to_train / BS * Epochs
	EPOCHS = 1

	"""
	以下是SGD使用办法，BS=1，靠人品
	"""
	LR          = 0.5
	BATCH_SIZE  = 1
	EPOCHS      = 1
	# MAX_ITERATIONS = 100000

	"""
	以下是Mini-Batch SGD使用办法，BS=10,靠人品
	尝试：LR>=0.1, BS<=200,EPOCHS>=1
	"""
	LR          = 0.25
	BATCH_SIZE  = 10
	EPOCHS      = 5

	"""
	如果追求准确率，建议：LR<=0.03, BS>=500, EPOCHS>=300
	可以得到较好的结果：[TEST RESULT] err: [302/2000], acc: 84.90%
	"""
	LR          = 0.03
	BATCH_SIZE  = 100
	EPOCHS      = 20

	"""
	大乱斗冠军参数（但线上并不够理想，在io、运算和算法上还有很大优化空间）
	"""
	LR = 0.01
	BATCH_SIZE = 10
	EPOCHS = 10


	main()

