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
		raise NotImplementedError

	def forward_backward(self, x, y):
		raise NotImplementedError

	def join(self):
		raise NotImplementedError



if __name__ == '__main__':
	pass


###
