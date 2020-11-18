"""
Train a distributed model.

Usage
-----
$ python training_script.py [model_type] [dataset_type]
	model_type options:
		1: <class 'code.models.basic_distributed_model.BasicDistributedModel'>
		2: <class 'code.models.prof_basic_distributed_model.ProfBasicDistributedModel'>
		3: <class 'code.models.gpipe_model.GpipeModel'>
		4: <class 'code.models.prof_gpipe_model.ProfGpipeModel'>
		5: <class 'code.models.refinement_model.RefinementModel'>
		6: <class 'code.models.prof_refinement_model.ProfRefinementModel'>
	dataset_type options:
		1: XOR dataset
		2: MNIST dataset

"""
__date__ = "November 2020"

import sys
import torch

from code.data.xor_data import get_xor_loaders, plot_model_xor_predictions
from code.data.mnist_data import get_mnist_loaders, plot_model_mnist_predictions

from code.models.basic_distributed_model import BasicDistributedModel
from code.models.prof_basic_distributed_model import ProfBasicDistributedModel
from code.models.gpipe_model import GpipeModel
from code.models.prof_gpipe_model import ProfGpipeModel
from code.models.refinement_model import RefinementModel
from code.models.prof_refinement_model import ProfRefinementModel


BATCH_SIZE = 25
NUM_BATCHES = 8
DSET_SIZE = BATCH_SIZE * NUM_BATCHES

MODEL_OPTIONS = {
	1: BasicDistributedModel,
	2: ProfBasicDistributedModel,
	3: GpipeModel,
	4: ProfGpipeModel,
	5: RefinementModel,
	6: ProfRefinementModel,
}

DATASET_OPTIONS = {
	1: {
		'name': "XOR dataset",
		'loader_func': get_xor_loaders,
		'plot_func': plot_model_xor_predictions,
		'net_dims': [[2,10,10], [10,10,10], [10,10,1]],
	},
	2: {
		'name': "MNIST dataset",
		'loader_func': get_mnist_loaders,
		'plot_func': plot_model_mnist_predictions,
		'net_dims': [[392,64,32], [32,32,32], [32,64,392]],
	}
}

USAGE_STR = "Usage\n-----\n"
USAGE_STR += "$ python training_script.py [model_type] [dataset_type]\n"
USAGE_STR += "\tmodel_type options:\n"
for option in sorted(MODEL_OPTIONS.keys()):
	USAGE_STR += '\t\t' + str(option) + ': ' + str(MODEL_OPTIONS[option]) +'\n'
USAGE_STR += "\tdataset_type options:\n"
for option in sorted(DATASET_OPTIONS.keys()):
	USAGE_STR += '\t\t' + str(option) + ': ' + DATASET_OPTIONS[option]['name']
	USAGE_STR += '\n'



def train_loop(model, loaders, epochs=100, test_freq=10):
	"""
	Iterate through the data and take gradient steps.

	Note: the `wait` parameter is for telling the GPipe models whether to wait
	for more batches before running the backward pass.

	Parameters
	----------
	model : DistributedModel
	loader : dict mapping 'train' and 'test' to DataLoaders
	epochs : int, optional
	test_freq : int, optional
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
	if len(sys.argv) != 3:
		print(USAGE_STR)
		exit(1)
	try:
		model_num = int(sys.argv[1])
		dataset_num = int(sys.argv[2])
	except ValueError:
		print(USAGE_STR)
		exit(1)
	if model_num not in MODEL_OPTIONS or dataset_num not in DATASET_OPTIONS:
		print(USAGE_STR)
		exit(1)

	# Get DataLoaders.
	loader_func = DATASET_OPTIONS[dataset_num]['loader_func']
	loaders = loader_func(n_samples=DSET_SIZE, batch_size=BATCH_SIZE)

	# Make the model.
	net_dims = DATASET_OPTIONS[dataset_num]['net_dims']
	model = MODEL_OPTIONS[model_num](net_dims, NUM_BATCHES)

	# Train.
	train_loop(model, loaders)

	# Plot.
	plot_func = DATASET_OPTIONS[dataset_num]['plot_func']
	plot_func(model, n_samples=DSET_SIZE, seed=42)

	# Clean up.
	model.join()



###
