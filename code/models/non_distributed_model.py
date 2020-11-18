"""
A non-distributed model for comparing gradients.

"""
__date__ = "November 2020"


import torch

from .utils import make_dense_net



class NonDistributedModel(torch.nn.Module):


	def __init__(self, net_dims, seed=False):
		"""
		Parameters
		----------
		net_dims : list of list of int
		seed : bool, optional
		"""
		super(NonDistributedModel, self).__init__()
		assert len(net_dims) > 1
		new_net_dims = [j for j in net_dims[0]]
		for i in range(1,len(net_dims)):
			new_net_dims += [j for j in net_dims[i][1:]]
		# Make a network.
		self.net = make_dense_net(new_net_dims, include_last_relu=False, \
				seed=seed)


	def forward(self, x):
		"""Do a forward pass through the network."""
		return self.net(x)


	def forward_backward(self, x, y, return_grads=False):
		"""Do a forward and backward pass."""
		y_pred = self.forward(x)
		loss = torch.mean(torch.pow(y - y_pred, 2))
		loss.backward()
		if return_grads:
			return self.net[0].bias.grad


if __name__ == '__main__':
	pass


###
