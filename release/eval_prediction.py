# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/20 6:45
# @Author	   : Mark Shawn
# @Email		: shawninjuly@gmail.com
# ------------------------------------
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


def check_answer():
	from collections import defaultdict

	info = defaultdict(int)
	with open(test_predict_path, "r") as f:
		pred_y = [int(i) for i in f]
	with open(test_answer_path, "r") as f:
		target_y = [int(i) for i in f]
	assert len(pred_y) == len(target_y)
	for i, j in zip(pred_y, target_y):
		info[(i, j)] += 1

	print("{:12s}\t{:8d}\t{:8d}".format("Pred\\Target", 0, 1))
	print("{:12s}\t{:8d}\t{:8d}".format('0', info[(0,0)], info[(0,1)]))
	print("{:12s}\t{:8d}\t{:8d}".format('1', info[(1,0)], info[(1,1)]))

	n_errs = info[(0,1)]+info[(1,0)]
	print("err: [{}/{}], acc: {:.4f}%".format(
		n_errs, len(pred_y), (1-n_errs/len(pred_y))*100
	))


if __name__ == '__main__':
	check_answer()