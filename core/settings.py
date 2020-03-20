# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/20 21:05
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
import os


DATA_DIR = os.path.join(__file__, "../../../R0/data")
if not os.path.exists(DATA_DIR):
	raise NotImplementedError("未找到您的数据文件夹: {}".format(os.path.abspath(DATA_DIR)))