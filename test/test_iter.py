# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/21 22:29
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------


class DataLoader:

	def __init__(self):
		self.x = list(range(5))

	def __iter__(self):
		print("Hello")
		for i in self.x:
			yield i


if __name__ == '__main__':
	dl = DataLoader()
	for i in dl:
		print(i)
