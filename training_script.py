"""
Train a distributed model.

Usage
=====
$ python training_script.py [model_type] [dataset_type] [cpu_affinity] [model_size] [batch_size] [n_batch] [epochs] [save_fn]
	model_type options:
		1: BasicDistributedModel
		2: BasicDistributedModel (with profiling)
		3: GpipeModel
		4: GpipeModel (with profiling)
		5: RefinementModel
		6: RefinementModel (with profiling)
	dataset_type options:
		1: XOR dataset
		2: MNIST dataset
	cpu_affinity options:
		1: no affinities
		2: round robin affinities (Linux only)
	model_size options:
		1: small
		2: large

"""
__date__ = "November 2020"

import os
import sys
import torch
torch.manual_seed(0)

from code.data.xor_data import get_xor_loaders, plot_model_xor_predictions
from code.data.mnist_data import get_mnist_loaders, plot_model_mnist_predictions

from code.models.basic_distributed_model import BasicDistributedModel
from code.models.prof_basic_distributed_model import ProfBasicDistributedModel
from code.models.gpipe_model import GpipeModel
from code.models.prof_gpipe_model import ProfGpipeModel
from code.models.refinement_model import RefinementModel
from code.models.prof_refinement_model import ProfRefinementModel



MODEL_OPTIONS = {
	1: {
		'name': 'BasicDistributedModel',
		'model': BasicDistributedModel,
	},
	2: {
		'name': 'BasicDistributedModel (with profiling)',
		'model': ProfBasicDistributedModel,
	},
	3: {
		'name': 'GpipeModel',
		'model': GpipeModel,
	},
	4: {
		'name': 'GpipeModel (with profiling)',
		'model': ProfGpipeModel,
	},
	5: {
		'name': 'RefinementModel',
		'model': RefinementModel,
	},
	6: {
		'name': 'RefinementModel (with profiling)',
		'model': ProfRefinementModel,
	},
}

DATASET_OPTIONS = {
	1: {
		'name': "XOR dataset",
		'loader_func': get_xor_loaders,
		'plot_func': plot_model_xor_predictions,
		'net_dims': {
			'small': [[2,10,10], [10,10,10], [10,10,1]],
			'large': [[2,128,128], [128,128,128], [128,128,128], [128,128,1]],
		},
	},
	2: {
		'name': "MNIST dataset",
		'loader_func': get_mnist_loaders,
		'plot_func': plot_model_mnist_predictions,
		'net_dims': {
			'small': [[392,64,32], [32,32,32], [32,64,392]],
			'large': [ \
				[392,128,128],
				[128,128,128],
				[128,128,128],
				[128,128,128],
				[128,128,128],
				[128,128,128],
				[128,128,392],
			],
		}
	}
}

CPU_AFFINITY_OPTIONS = {
	1: {
		'name': 'no affinities',
		'value': False,
	},
	2: {
		'name': 'round robin affinities (Linux only)',
		'value': True,
	},
}

MODEL_SIZE_OPTIONS = {
	1: 'small',
	2: 'large',
}

USAGE_STR = "Usage\n=====\n"
USAGE_STR += "$ python training_script.py [model_type] [dataset_type] "
USAGE_STR += "[cpu_affinity] [model_size] [batch_size] [n_batch] [epochs] "
USAGE_STR += "[save_fn]\n\tmodel_type options:\n"
for option in sorted(MODEL_OPTIONS.keys()):
	USAGE_STR += '\t\t' + str(option) + ': ' + MODEL_OPTIONS[option]['name']
	USAGE_STR += '\n'
USAGE_STR += "\tdataset_type options:\n"
for option in sorted(DATASET_OPTIONS.keys()):
	USAGE_STR += '\t\t' + str(option) + ': ' + DATASET_OPTIONS[option]['name']
	USAGE_STR += '\n'
USAGE_STR += "\tcpu_affinity options:\n"
for option in sorted(CPU_AFFINITY_OPTIONS.keys()):
	USAGE_STR += '\t\t' + str(option) + ': '
	USAGE_STR += CPU_AFFINITY_OPTIONS[option]['name'] + '\n'
USAGE_STR += "\tmodel_size options:\n"
for option in sorted(MODEL_SIZE_OPTIONS.keys()):
	USAGE_STR += '\t\t' + str(option) + ': '
	USAGE_STR += MODEL_SIZE_OPTIONS[option] + '\n'

CPU_AFF_ERROR_MSG = "CPU affinity can only be set on Linux systems!\n\t"
CPU_AFF_ERROR_MSG += "Found system: " + os.uname().sysname


def parse_args():
	"""Parse command line arguments."""
	# Parse arguments.
	if len(sys.argv) != 9:
		print(USAGE_STR)
		exit(1)
	try:
		model_num = int(sys.argv[1])
		dataset_num = int(sys.argv[2])
		cpu_affinity_option = int(sys.argv[3])
		model_size_option = int(sys.argv[4])
		batch_size = int(sys.argv[5])
		num_batches = int(sys.argv[6])
		epochs = int(sys.argv[7])
		save_fn = sys.argv[8]
	except ValueError:
		print(USAGE_STR)
		exit(1)
	assert batch_size > 0
	assert num_batches > 0
	if (model_num not in MODEL_OPTIONS) or \
				(dataset_num not in DATASET_OPTIONS) or \
				(cpu_affinity_option not in CPU_AFFINITY_OPTIONS) or \
				(model_size_option not in MODEL_SIZE_OPTIONS):
		print(USAGE_STR)
		exit(1)
	# Make sure we're on a Linux system if CPU affinities are set.
	cpu_affinity = CPU_AFFINITY_OPTIONS[cpu_affinity_option]['value']
	if cpu_affinity:
		assert os.uname().sysname == 'Linux', CPU_AFF_ERROR_MSG
	model_size = MODEL_SIZE_OPTIONS[model_size_option]
	return model_num, dataset_num, cpu_affinity, model_size, batch_size, \
			num_batches, epochs, save_fn


def train_loop(model, loaders, num_batches, epochs=200, test_freq=10):
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
			wait = int(batch == num_batches-1)
			model.forward_backward(features, targets, wait=wait)
		if epoch % test_freq == 0:
			for batch, (features, targets) in enumerate(loaders['test']):
				wait = int(batch == num_batches-1)
				model.test_forward(features, targets, wait=wait)



if __name__ == '__main__':
	# Parse arguments.
	model_num, dataset_num, cpu_affinity, model_size, batch_size, num_batches, \
			epochs, save_fn = parse_args()
	dset_size = num_batches * batch_size

	# Get DataLoaders.
	loader_func = DATASET_OPTIONS[dataset_num]['loader_func']
	loaders = loader_func(n_samples=dset_size, batch_size=batch_size)

	# Make the model.
	net_dims = DATASET_OPTIONS[dataset_num]['net_dims'][model_size]
	model = MODEL_OPTIONS[model_num]['model']( \
			net_dims,
			num_batches,
			cpu_affinity=cpu_affinity,
			save_fn=save_fn,
	)

	# Train.
	print("Training", MODEL_OPTIONS[model_num]['name']+'...')
	train_loop(model, loaders, num_batches, epochs=epochs)

	# Plot.
	plot_func = DATASET_OPTIONS[dataset_num]['plot_func']
	plot_func(model, n_samples=dset_size, seed=42)

	# Clean up.
	model.join()



###
