# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/20 5:16
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------


import pickle
import pandas as pd

eval_data = pickle.load(open("eval_data.pkl", "rb"))
df = pd.DataFrame(eval_data)


def punish_rate(x):
	if x > 1:
		x /= 100
	if x > 0.95:
		return 1
	elif x >= 0.9:
		return 1.2
	elif x >= 0.8:
		return 1.5
	elif x >= 0.7:
		return 2
	else:
		return 100000

df["score"] = df.acc.apply(punish_rate) * df.TRAIN_TIME
df.sort_values("score", inplace=True)
