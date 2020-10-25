"""
A simple example, now with paralllelism.

TO DO
-----
- Make sure this is working correctly.
- Wrap everything in torch.nn.Module
"""
__date__ = "October 2020"

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import torch
torch.autograd.set_detect_anomaly(True) # useful for debugging

BATCH_DIM = 100
VERBOSE = False


class NoOp(torch.nn.Module):
	"""Useful for caching gradients."""

	def __init__(self, dim_1, dim_2):
		super(NoOp, self).__init__()
		self.bias = torch.zeros(dim_1, dim_2, requires_grad=True)

	def forward(self, x):
		return self.bias + x

	def zero(self):
		"""Return to zeros."""
		with torch.no_grad():
			self.bias.data = torch.zeros_like(self.bias.data)


def get_xor_training_data(n_samples=BATCH_DIM, seed=42, std_dev=0.5):
	"""
	Get some training data for a simple XOR task.

	Parameters
	----------
	n_samples : int, optional
		Number of samples.
	seed : int, optional
		Numpy random seed.

	Returns
	-------
	features : torch.Tensor
		shape: [n_samples,2]
	targets : torch.Tensor
		shape: [n_samples,1]
	"""
	np.random.seed(seed)
	cluster_ids = np.random.randint(4, size=n_samples)
	features = std_dev * np.random.normal(size=(n_samples,2))
	np.random.seed(None)
	features[cluster_ids == 0,0] -= 1.0
	features[cluster_ids == 0,1] -= 1.0
	features[cluster_ids == 1,0] -= 1.0
	features[cluster_ids == 1,1] += 1.0
	features[cluster_ids == 2,0] += 1.0
	features[cluster_ids == 2,1] -= 1.0
	features[cluster_ids == 3,0] += 1.0
	features[cluster_ids == 3,1] += 1.0
	targets = np.zeros((n_samples,1))
	targets[cluster_ids == 1] = 1.0
	targets[cluster_ids == 2] = 1.0
	# Convert to torch tensors.
	features = torch.tensor(features).to(torch.float)
	targets = torch.tensor(targets).to(torch.float)
	return features, targets


def make_dense_net(layer_dims, include_no_op=True, include_last_relu=True):
	"""
	Make a network with ReLUs and dense layers.

	Parameters
	----------
	layer_dims : list of int
		List of layer dimensions
	"""
	assert len(layer_dims) >= 2
	layer_list = []
	if include_no_op:
		layer_list.append(NoOp(BATCH_DIM, layer_dims[0]))
	for i in range(len(layer_dims)-1):
		layer_list.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
		if include_last_relu or i != len(layer_dims)-2:
			layer_list.append(torch.nn.ReLU())
	return torch.nn.Sequential(*layer_list).to(torch.float)


def mp_target_func(net_dims, parent_conn, child_conn, target_conn):
	"""
	Spawn a torch.nn.Module

	Parameters
	----------
	net_dims : list of int
		List of layer dimensions
	parent_conn : multiprocessing.connection.Connection
	child_conn : multiprocessing.connection.Connection or None
	target_conn : multiprocessing.connection.Connection or None
	"""
	if VERBOSE:
		print("Process created with pid:", os.getpid(), "\n\tdims:", net_dims)
	# Make a network.
	net = make_dense_net(net_dims, include_last_relu=(target_conn is None))
	# Make an optimizer.
	optimizer = torch.optim.Adam(net.parameters())

	epoch = 0
	while True:
		epoch += 1
		# Receive data from our parent.
		data = parent_conn.recv()

		# Return on None.
		if data is None:
			if child_conn is not None:
				child_conn.send(None)
			return

		if VERBOSE:
			print("process", os.getpid(), "received data:", data.shape, data.requires_grad)

		# Zero gradients.
		optimizer.zero_grad()

		# Make a forward pass.
		output = net(data)
		if child_conn is None:
			# If we're the last Module, calculate a loss.
			target = target_conn.recv() # Receive the targets.
			if VERBOSE:
				print("process", os.getpid(), "received target:", target.shape)
			loss = torch.mean(torch.pow(output - target, 2))
			if epoch % 100 == 0:
				print("epoch:", epoch ,"loss:", loss.item())
			loss.backward()
		else:
			# Otherwise, pass the output to our children.
			if VERBOSE:
				print("process", os.getpid(), "sending data:", output.shape)
			child_conn.send(output.detach())
			grads = child_conn.recv()
			if VERBOSE:
				print("process", os.getpid(), "received grads:", grads.shape)
			# Fake a loss with the correct gradients.
			loss = torch.sum(output * grads)
			loss.backward()
			# output.backward(gradient=grads) # should do the same thing as above
		# Pass gradients back.
		parent_conn.send(net[0].bias.grad)
		# Update this module's parameters.
		optimizer.step()
		# And zero out the NoOp.
		net[0].zero()


def training_run(model, epochs=1001, print_freq=100):
	"""
	Train the model.

	Parameters
	----------
	model : torch.nn.Module
		Network to train
	epochs : int, optional
		Training epochs
	print_freq : int, optional
	"""
	features, targets = get_xor_training_data()
	optimizer = torch.optim.Adam(model.parameters())
	for epoch in range(epochs):
		optimizer.zero_grad()
		prediction = model(features)
		loss = torch.mean(torch.pow(prediction - targets, 2))
		if epoch % print_freq == 0:
			print("epoch", epoch, "loss", loss.item())
		loss.backward()
		optimizer.step()


def plot_model_predictions(model, max_x=3, grid_points=40):
	""" """
	# Make a grid of points.
	x = torch.linspace(-max_x,max_x,grid_points).to(torch.float)
	y = torch.linspace(-max_x,max_x,grid_points).to(torch.float)
	grid_x, grid_y = torch.meshgrid(x, y)
	grid = torch.stack([grid_y,grid_x], dim=-1)
	# Have the model predict outputs for each grid point.
	with torch.no_grad():
		prediction = model(grid).detach().cpu().numpy()
	prediction = prediction.reshape(grid_points,grid_points)
	# Plot predictions.
	plt.imshow(prediction, extent=[-max_x,max_x,-max_x,max_x], \
		interpolation='bicubic')
	plt.colorbar()
	# Show training data.
	features, targets = get_xor_training_data()
	features, targets = features.numpy(), targets.numpy().flatten()
	idx_0 = np.argwhere(targets == 0)
	idx_1 = np.argwhere(targets == 1)
	plt.scatter(features[idx_0,0].flatten(), features[idx_0,1].flatten(), c='r')
	plt.scatter(features[idx_1,0].flatten(), features[idx_1,1].flatten(), c='k')
	plt.ylabel('Feature 1')
	plt.xlabel('Feature 2')
	plt.title('Network Predictions')
	plt.savefig('temp.pdf')



if __name__ == '__main__':
	# Get training data.
	features, targets = get_xor_training_data()
	# Define network dimensions.
	net_1_dims = [2,10,10]
	net_2_dims = [10,10,1]
	# Make the pipes.
	conn_1r, conn_1s = mp.Pipe() # Parent to first module
	conn_2r, conn_2s = mp.Pipe() # First module to second
	conn_3r, conn_3s = mp.Pipe() # Second module to parent
	# Spawn the processes.
	p1 = mp.Process(target=mp_target_func, args=(net_1_dims, conn_1r, conn_2s, None))
	p1.start()
	p2 = mp.Process(target=mp_target_func, args=(net_2_dims, conn_2r, None, conn_3r))
	p2.start()
	# Send in the features.
	for epoch in range(1000):
		conn_1s.send(features)
		conn_3s.send(features)
		_ = conn_1s.recv() # Gradients come back.
	conn_1s.send(None) # This is the kill signal.
	p1.join()
	p2.join()



###
