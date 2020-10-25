"""
A simple example showing how we can chain together torch Modules using the
torch Sequntial abstraction. All the gradients are handled automatically.

Note that torch.nn.Sequential subclasses torch.nn.Module

"""
__date__ = "October 2020"

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True) # useful for debugging



def get_xor_training_data(n_samples=100, seed=42, std_dev=0.5):
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


def make_dense_net(layer_dims):
	"""
	Make a network with ReLUs and dense layers.

	Parameters
	----------
	layer_dims : list of int
		List of layer dimensions
	"""
	assert len(layer_dims) >= 2
	layer_list = []
	for i in range(len(layer_dims)-1):
		layer_list.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
		if i != len(layer_dims)-2:
			layer_list.append(torch.nn.ReLU())
	return torch.nn.Sequential(*layer_list).to(torch.float)


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
		loss = torch.mean(torch.pow(prediction - targets,2))
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
	# Make two small networks.
	net_1 = make_dense_net([2,10,10])
	net_2 = make_dense_net([10,10,1])
	# Combine the networks.
	combo_model = torch.nn.Sequential(net_1, net_2)
	# Train.
	training_run(combo_model)
	# Plot.
	plot_model_predictions(combo_model)


###
