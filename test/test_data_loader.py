import unittest
from MachineLarning_Numpy_CodeCraft2020.core.common import *
from MachineLarning_Numpy_CodeCraft2020.core.dataloaders import DataLoader


class MyTestCase(unittest.TestCase):

	def test_something(self):
		train_data_loader = DataLoader(batch_size=1000, shuffle=True)
		train_data_loader.load_XY(train_data_path)
		for X, y in train_data_loader:
			self.assertEqual(X.shape, (1000, 1000))
			self.assertEqual(y.shape, (1000, ))
			self.assertEqual(y.dtype, "int")

		test_data_loader = DataLoader()
		test_data_loader.load_X(test_data_path)
		test_data_loader.load_Y(test_answer_path)
		self.assertEqual(test_data_loader.data.shape, (2000, 1001))

if __name__ == '__main__':
	unittest.main()
