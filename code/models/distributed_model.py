"""
Define a distributed model abstract class.

"""
__date__ = "November 2020"


import torch



class DistributedModel(torch.nn.Module):
	"""DistributedModel abstract class"""

	def __init__(self):
		super(DistributedModel, self).__init__()


	def forward(self, x):
		"""
		Just the forward pass!

		Parameters
		----------
		x : torch.Tensor
			Features

		Returns
		-------
		y_est : torch.Tensor
			Estimated targets
		"""
		raise NotImplementedError


	def forward_backward(self, x, y):
		"""
		Perform both the forward and backward passes.

		Parameters
		----------
		x : torch.Tensor
			Features
		y : torch.Tensor
			Targets
		"""
		raise NotImplementedError


	def join(self):
		"""Join all the processes."""
		raise NotImplementedError



if __name__ == '__main__':
	pass


###
