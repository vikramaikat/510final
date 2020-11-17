"""
Define a No-Op Pytorch Module that's useful for caching gradients.

"""
__date__ = "November 2020"

import torch



class NoOp(torch.nn.Module):
	"""Useful for caching gradients."""

	def __init__(self, dim):
		super(NoOp, self).__init__()
		self.dim = dim
		self.bias = None

	def forward(self, x):
		self.bias = torch.zeros(x.shape[0], self.dim, requires_grad=True)
		return self.bias + x

	def zero(self):
		"""Zero out the gradients."""
		self.bias.grad.fill_(0.0)



if __name__ == '__main__':
	pass


###
