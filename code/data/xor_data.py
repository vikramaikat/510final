"""
Toy data: noisy XOR classification.

"""
__date__ = "November 2020"

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
import torch



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


def get_xor_loaders(n_samples=100, batch_size=25):
	"""Get XOR data test/train loaders."""
	# Make train loader.
	features, targets = get_xor_training_data(n_samples=n_samples, seed=42)
	dset = torch.utils.data.TensorDataset(features, targets)
	train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, \
			shuffle=True)
	# Make test loader.
	features, targets = get_xor_training_data(n_samples=n_samples, seed=43)
	dset = torch.utils.data.TensorDataset(features, targets)
	test_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, \
			shuffle=True)
	return {'train': train_loader, 'test': test_loader}


def plot_model_xor_predictions(model, max_x=3, grid_points=80, \
	img_fn='xor_predictions.pdf', n_samples=100, seed=42):
	"""
	Plot the model predictions on the XOR dataset.

	Parameters
	----------
	model : DistributedModel
	max_x : float, optional
	grid_points : int, optional
	img_fn : str, optional
	"""
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
	plt.imshow(prediction, extent=[-max_x,max_x,-max_x,max_x], \
			interpolation='bicubic', cmap='viridis', vmin=vmin, vmax=vmax)
	plt.colorbar()
	# Show training data.
	features, targets = get_xor_training_data(n_samples=n_samples, seed=seed)
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
	# Save and close.
	plt.savefig(img_fn)
	plt.close('all')



if __name__ == '__main__':
	pass


###
