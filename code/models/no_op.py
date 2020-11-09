"""
Define a No-Op Pytorch Module that will be useful for caching gradients.

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
		"""Return to None."""
		with torch.no_grad():
			self.bias.grad = 0.0 * self.bias.grad



if __name__ == '__main__':
	pass


###
