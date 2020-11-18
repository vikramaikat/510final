"""
Toy data: predict one half of MNIST digits from the other.

"""
__date__ = "November 2020"

import matplotlib.pyplot as plt
import numpy as np
import torch

N_PLOT_IMGS = 4


def get_mnist_training_data(n_samples=100, seed=42, fn='mnist_test.csv', \
	train=True):
	"""
	Get some training data for an MNIST prediction task.

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
	d = np.loadtxt(fn, delimiter=',')
	assert d.shape[0] >= 2*n_samples
	np.random.seed(seed)
	perm = np.random.permutation(d.shape[0])
	np.random.seed(None)
	if train:
		perm = perm[:n_samples]
	else:
		perm = perm[n_samples:2*n_samples]
	d = d[perm,1:] / 255.0
	# Convert to torch tensors.
	idx = d.shape[1] // 2
	features = torch.tensor(d[:,:idx]).to(torch.float)
	targets = torch.tensor(d[:,idx:]).to(torch.float)
	return features, targets


def get_mnist_loaders(n_samples=100, batch_size=25):
	"""Get MNIST data test/train loaders."""
	# Make train loader.
	features, targets = get_mnist_training_data(n_samples=n_samples, seed=42)
	dset = torch.utils.data.TensorDataset(features, targets)
	train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, \
			shuffle=True)
	# Make test loader.
	features, targets = get_mnist_training_data(n_samples=n_samples, seed=43)
	dset = torch.utils.data.TensorDataset(features, targets)
	test_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, \
			shuffle=True)
	return {'train': train_loader, 'test': test_loader}


def plot_model_mnist_predictions(model, img_fn='mnist_predictions.pdf', \
	n_samples=100, seed=42, train=True, data_fn='mnist_test.csv'):
	"""
	Plot the model predictions on the MNIST dataset.

	Parameters
	----------
	model : DistributedModel
	img_fn : str, optional
	n_samples : int, optional
	seed : int, optional
	train : bool, optional
	"""
	# Get data.
	features, targets = get_mnist_training_data(n_samples=n_samples, seed=seed,\
			train=train, fn=data_fn)
	features, targets = features[:N_PLOT_IMGS], targets[:N_PLOT_IMGS]
	# Have the model predict targets.
	with torch.no_grad():
		prediction = model(features)
		img = torch.cat([features,prediction], dim=1).detach().cpu().numpy()
	# Reshape things.
	img = img.reshape(N_PLOT_IMGS*28,28)
	plt.imshow(img, aspect='equal', cmap='gray', vmin=-0.1, vmax=1.1)
	plt.axis('off')
	# Save and close.
	plt.savefig(img_fn)
	plt.close('all')



if __name__ == '__main__':
	pass


###
