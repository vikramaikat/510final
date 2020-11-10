"""
Train a distributed model.

"""
__date__ = "November 2020"

import sys
import torch

from code.data.xor_data import get_xor_training_data, plot_model_predictions
from code.models.gpipe_model import GpipeModel
from code.models.basic_distributed_model import BasicDistributedModel


BATCH_SIZE = 25
NUM_BATCHES = 4
DSET_SIZE = BATCH_SIZE * NUM_BATCHES
NET_DIMS = [[2,10,10], [10,10,1]]
MODEL_OPTIONS = {
	1: BasicDistributedModel,
	2: GpipeModel,
}
USAGE_STR = "$ python training_script.py [model_type]\n\tmodel_type options:\n"
for option in sorted(MODEL_OPTIONS.keys()):
	USAGE_STR += '\t' + str(option) + ': ' + str(MODEL_OPTIONS[option]) +'\n'



def train_loop(model, loader, epochs=1000):
	"""
	Iterate through the data and take gradient steps.

	Parameters
	----------
	model : DistributedModel
	loader : DataLoader
	epochs : int, optional
	"""
	for epoch in range(epochs):
		for batch, (features, targets) in enumerate(loader):
			wait_num = NUM_BATCHES * int(batch == NUM_BATCHES-1)
			model.forward_backward(features, targets, wait_num=wait_num)



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
	# Get training data.
	features, targets = get_xor_training_data(n_samples=DSET_SIZE)
	# Make a Dataset.
	dset = torch.utils.data.TensorDataset(features, targets)
	# Make a DataLoader.
	loader = torch.utils.data.DataLoader(dset, batch_size=BATCH_SIZE, \
			shuffle=True)
	# Make the model.
	model = MODEL_OPTIONS[model_num](NET_DIMS, NUM_BATCHES)
	# Send in the features.
	train_loop(model, loader)
	# Plot.
	plot_model_predictions(model)
	# Clean up.
	model.join()



###
