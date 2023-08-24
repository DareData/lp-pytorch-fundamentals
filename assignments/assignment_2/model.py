# Here, you must create the model class. It should be a subclass of torch.nn.Module.

import torch
import torch.nn as nn

class HeartAttackModel(nn.Module):
	def __init__(self):
		# TODO: Implement this method. It should define the following layers:
		# 1. A fully connected (Linear) layer with N(umber of features) inputs and 16 outputs
		# 2. A fully connected (Linear) layer with 16 inputs and 8 outputs
		# 3. A fully connected (Linear) layer with 8 inputs and 1 output
		# 4. A ReLU activation function

		# Note: Don't forget to call the super().__init__() method!
		pass

	def forward(self, x):
		# TODO: Implement this method.
		# Don't forget that the forward pass should apply the ReLU activation function
		# in between the linear layers, but not after the last one.
		pass