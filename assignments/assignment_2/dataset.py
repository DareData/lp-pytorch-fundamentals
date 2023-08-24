# Here, you must create your Dataset class

import torch
from torch.utils.data import Dataset
import pandas as pd

class HeartAttackDataset(Dataset):
	def __init__(self):
		# TODO: Implement this method. It should load the data from the csv file,
		# process the data as required  and define the 
		# self.features and self.labels variables, as torch tensors.
		self.features = None # dont forget the float32 dtype
		self.labels = None # dont forget the float32 dtype
		pass

	def __len__(self):
		# TODO: Implement this method. It should return the length of the dataset.
		# It's a required method for the Dataset class to work.
		pass

	def __getitem__(self, idx):
		# TODO: Implement this method. It should return a tuple (x, y) where x is a
		# torch tensor with the sample features and y is a torch tensor with the
		# sample label.
		pass