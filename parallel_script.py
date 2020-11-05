"""
A simple example, now with paralllelism.

TO DO
-----
- Make sure this is working correctly.
- Add batches, gpipe microbatches.
"""
__date__ = "October 2020"

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import torch
torch.autograd.set_detect_anomaly(True) # useful for debugging

BATCH_DIM = 100


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
		self.bias = None


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


def get_gaussian_training_data(n_samples=BATCH_DIM, seed=42):
	"""
		Get some training data for a linearly separable task.

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
	features = np.zeros((n_samples, 2))
	f1 = np.random.multivariate_normal([-1,-1], [[.1, .3], [.3,.1]], n_samples//2)
	f2 = np.random.multivariate_normal([1,1], [[.3, .1], [.1,.3]], n_samples//2)
	features = np.concatenate((f1, f2))
	t1 = np.zeros((n_samples//2, 1))
	t2 = np.ones((n_samples//2, 1))
	targets =  np.concatenate((t1, t2))
	# Convert to torch tensors.
	features = torch.tensor(features).to(torch.float)
	targets = torch.tensor(targets).to(torch.float)
	return features, targets


def get_circle_training_data(n_samples=BATCH_DIM, seed=42, std_dev=0.5):
	"""
	TODO: make circle plot from http://playground.tensorflow.org/
	"""


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
		layer_list.append(NoOp(layer_dims[0]))
	for i in range(len(layer_dims)-1):
		layer_list.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
		if include_last_relu or i != len(layer_dims)-2:
			layer_list.append(torch.nn.ReLU())
	return torch.nn.Sequential(*layer_list).to(torch.float)


def mp_target_func(net_dims, parent_conn, child_conn, final_layer):
	"""
	Spawn a torch.nn.Module and wait for data.

	This is the meat of the parallelization.

	Parameters
	----------
	net_dims : list of int
		List of layer dimensions
	parent_conn : multiprocessing.connection.Connection
	child_conn : multiprocessing.connection.Connection
	"""
	# Make a network.
	net = make_dense_net(net_dims, include_last_relu=(not final_layer))
	# Make an optimizer.
	optimizer = torch.optim.Adam(net.parameters())
	# Enter main loop.
	epoch = 0
	loss_values = []
	while True:
		epoch += 1
		# Receive data from our parent.
		backwards_flag, data = parent_conn.recv()

		# Return on None.
		if data is None:
			plt.plot(loss_values)
			plt.title('Loss')
			plt.ylabel('Loss')
			plt.xlabel('Epoch')
			plt.savefig('batch_xor_loss.pdf')
			child_conn.send((None,None))
			return
		#print(data.shape)
		permutation = torch.randperm(data.size()[0])

		for i in range(0, data.size()[0], BATCH_DIM):

			# Zero gradients.
			optimizer.zero_grad()
			# Make a forward pass.
			indices = permutation[i:i+BATCH_DIM]
			batch_data = data[indices]
			output = net(batch_data)
			if backwards_flag: # If we're going to do a backwards pass:
				if final_layer:
					# If we're the last Module, calculate a loss.
					target = child_conn.recv() # Receive the targets.
					batch_target = target[indices]
					loss = torch.mean(torch.pow(output - batch_target, 2))
					loss_values.append(loss.item())
					if epoch % 100 == 0:
						print("epoch:", epoch ,"loss:", loss.item())
					loss.backward()
				else:
					# Otherwise, pass the output to our children.
					child_conn.send((True, output.detach()))
					grads = child_conn.recv()
					# Fake a loss with the correct gradients.
					loss = torch.sum(output * grads)
					loss.backward()
					# output.backward(gradient=grads) # should do the same thing as above
				# Pass gradients back.
				parent_conn.send(net[0].bias.grad)
				# Update this module's parameters.
				optimizer.step()
				# And zero out the NoOp layer.
				net[0].zero()
			else: # If we're just doing a forwards pass:
				# Just feed the activations to the child.
				child_conn.send((False, output.detach()))



class MPNet(torch.nn.Module):
	"""
	Network made up of distinct parameter partitions in different processes.

	Trained with standard backprop.

	TO DO
	-----
	* test
	* save/load models
	"""

	def __init__(self, net_dims):
		""" """
		super(MPNet, self).__init__()
		# Make all the pipes.
		self.pipes = [mp.Pipe() for _ in range(len(net_dims)+1)]
		# Make each network.
		self.processes = []
		for i, sub_net_dims in enumerate(net_dims):
			conn_1 = self.pipes[i][1]
			conn_2 = self.pipes[i+1][0]
			final_layer = (i == len(net_dims)-1)
			p = mp.Process( \
					target=mp_target_func,
					args=(sub_net_dims, conn_1, conn_2, final_layer),
			)
			self.processes.append(p)
			p.start()


	def forward(self, x):
		"""Just the forward pass!"""
		# Send features to the first parition.
		self.pipes[0][0].send((False, x))
		# Get predictions from last partition.
		_, prediction = self.pipes[-1][1].recv()
		return prediction


	def forward_backward(self, x, y):
		"""Both forward and backward passes."""
		# Send features to the first parition.
		self.pipes[0][0].send((True, x))
		# Send targets to the last parition.
		self.pipes[-1][1].send(y)
		_ = self.pipes[0][0].recv() # Wait for gradients to come back.


	def join(self):
		""" """
		# Send a kill signal.
		self.pipes[0][0].send((None,None))
		# Wait for it to come round.
		temp = self.pipes[-1][1].recv()
		assert temp == (None, None)
		# Join everything.
		for p in self.processes:
			p.join()

def plot_model_predictions(model, max_x=3, grid_points=40):
	"""Plot the model predictions on the XOR dataset."""
	# Make a grid of points.
	x = torch.linspace(-max_x,max_x,grid_points).to(torch.float)
	y = torch.linspace(-max_x,max_x,grid_points).to(torch.float)
	grid_x, grid_y = torch.meshgrid(x, y)
	grid = torch.stack([grid_y,grid_x], dim=-1)
	# Have the model predict outputs for each grid point.
	with torch.no_grad():
		prediction = model(grid).detach().cpu().numpy()
	prediction = prediction.reshape(grid_points,grid_points)
	vmin, vmax = np.min(prediction), np.max(prediction)
	# Plot predictions.
	plt.close()
	plt.imshow(prediction, extent=[-max_x,max_x,-max_x,max_x], \
		interpolation='bicubic', cmap='viridis', vmin=vmin, vmax=vmax)
	plt.colorbar()
	# Show training data.
	features, targets = get_xor_training_data()
	features, targets = features.numpy(), targets.numpy().flatten()
	idx_0 = np.argwhere(targets == 0)
	idx_1 = np.argwhere(targets == 1)
	cmap = get_cmap('viridis')
	color_0, color_1 = (0.0-vmin)/(vmax - vmin), (1.0-vmin)/(vmax - vmin)
	color_0, color_1 = np.array(cmap(color_0)), np.array(cmap(color_1))
	color_0, color_1 = color_0.reshape(1,-1), color_1.reshape(1,-1)
	plt.scatter(features[idx_0,0].flatten(), features[idx_0,1].flatten(), \
			c=color_1, edgecolors='k')
	plt.scatter(features[idx_1,0].flatten(), features[idx_1,1].flatten(), \
			c=color_0, edgecolors='k')
	plt.ylabel('Feature 1')
	plt.xlabel('Feature 2')
	plt.title('Network Predictions')
	plt.savefig('batch_xor_plot.pdf')



if __name__ == '__main__':
	# Get training data.
	features, targets = get_xor_training_data()
	# Define network dimensions.
	net_dims = [[2,10,10], [10,10,1]]
	# Make the model.
	model = MPNet(net_dims)
	# Send in the features.
	for epoch in range(1000):
		model.forward_backward(features, targets)
	# Plot.
	plot_model_predictions(model)
	# Clean up.
	model.join()



###
