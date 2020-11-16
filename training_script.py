"""
Train a distributed model.

"""
__date__ = "November 2020"

import sys
import torch

from code.data.xor_data import get_xor_training_data, plot_model_predictions
from code.models.gpipe_model import GpipeModel
from code.models.basic_distributed_model import BasicDistributedModel
from code.models.refinement_model import RefinementModel


BATCH_SIZE = 25
NUM_BATCHES = 4
DSET_SIZE = BATCH_SIZE * NUM_BATCHES
NET_DIMS = [[2,10,10], [10,10,10], [10,10,1]]
MODEL_OPTIONS = {
	1: BasicDistributedModel,
	2: GpipeModel,
	3: RefinementModel,
}
USAGE_STR = "Usage:\n"
USAGE_STR += "$ python training_script.py [model_type]\n\tmodel_type options:\n"
for option in sorted(MODEL_OPTIONS.keys()):
	USAGE_STR += '\t' + str(option) + ': ' + str(MODEL_OPTIONS[option]) +'\n'



def get_test_train_loaders():
	"""Get XOR data test/train loaders."""
	# Make train loader.
	features, targets = get_xor_training_data(n_samples=DSET_SIZE, seed=42)
	dset = torch.utils.data.TensorDataset(features, targets)
	train_loader = torch.utils.data.DataLoader(dset, batch_size=BATCH_SIZE, \
			shuffle=True)
	# Make test loader.
	features, targets = get_xor_training_data(n_samples=DSET_SIZE, seed=43)
	dset = torch.utils.data.TensorDataset(features, targets)
	test_loader = torch.utils.data.DataLoader(dset, batch_size=BATCH_SIZE, \
			shuffle=True)
	return {'train': train_loader, 'test': test_loader}



def train_loop(model, loaders, epochs=1000, test_freq=10):
	"""
	Iterate through the data and take gradient steps.

	Parameters
	----------
	model : DistributedModel
	loader : dict mapping 'train' and 'test' to DataLoaders
	epochs : int, optional
	"""
	for epoch in range(1, epochs+1):
		for batch, (features, targets) in enumerate(loaders['train']):
			wait = int(batch == NUM_BATCHES-1)
			model.forward_backward(features, targets, wait=wait)
		if epoch % test_freq == 0:
			for batch, (features, targets) in enumerate(loaders['test']):
				wait = int(batch == NUM_BATCHES-1)
				model.test_forward(features, targets, wait=wait)



if __name__ == '__main__':
	# Parse arguments.
	if len(sys.argv) != 2:
		print(USAGE_STR)
		exit(1)
	try:
		model_num = int(sys.argv[1])
	except ValueError:
		print(USAGE_STR)
		exit(1)
	if model_num not in MODEL_OPTIONS:
		print(USAGE_STR)
		exit(1)
	# Get DataLoaders.
	loaders = get_test_train_loaders()
	# Make the model.
	model = MODEL_OPTIONS[model_num](NET_DIMS, NUM_BATCHES)
	# Train.
	train_loop(model, loaders)
	# Plot.
	plot_model_predictions(model)
	# Clean up.
	model.join()



###
