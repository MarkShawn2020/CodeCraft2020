# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/19 23:54
# @Author	   : Mark Shawn
# @Email		: shawninjuly@gmail.com
# ------------------------------------


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
	data_dir = '/data'
	train_data_path = "/data/train_data.txt"
	test_data_path = "/data/test_data.txt"
	test_answer_path = "/data/answer.txt"
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
