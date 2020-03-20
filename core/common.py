# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/19 23:54
# @Author	   : Mark Shawn
# @Email		: shawninjuly@gmail.com
# ------------------------------------
from .settings import DATA_DIR

class lazy:
	def __init__(self, func):
		self.func = func

	def __get__(self, instance, cls):
		val = self.func(instance)
		setattr(instance, self.func.__name__, val)
		return val

import os, sys
if "win" in sys.platform:
	train_data_path = os.path.join(DATA_DIR, "train_data.txt")
	test_data_path = os.path.join(DATA_DIR, "test_data.txt")
	test_answer_path = os.path.join(DATA_DIR, "answer.txt")
	test_predict_path = os.path.join(DATA_DIR, "result.txt")
else:
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
