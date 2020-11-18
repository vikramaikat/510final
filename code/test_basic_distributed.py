"""
Test suite for distributed models.

Contains
--------
* test_basic_distributed_model_activations
* test_basic_distributed_model_gradients

"""
__date__ = "November 2020"


import numpy as np
import torch
import unittest

from code.models.basic_distributed_model import BasicDistributedModel
from code.models.non_distributed_model import NonDistributedModel


NUM_BATCHES = 4
BATCH_SIZE = 25
NET_DIMS = [[2,10,10], [10,10,1]]



class BasicDistributedModelTestCases(unittest.TestCase):

	def test_basic_distributed_model_activations(self):
		# Define the inputs.
		input_1 = torch.randn(BATCH_SIZE, NET_DIMS[0][0])
		input_2 = input_1[:]

		# Make the ground truth model and get activations.
		model_1 = NonDistributedModel(NET_DIMS, seed=True)
		with torch.no_grad():
			output_1 = model_1(input_1).detach().cpu().numpy()

		# Make the distributed model and get activations.
		model_2 = BasicDistributedModel(NET_DIMS, NUM_BATCHES, seed=True)
		with torch.no_grad():
			output_2 = model_2(input_2).detach().cpu().numpy()
		model_2.join()

		self.assertTrue( \
				np.allclose(output_1, output_2),
				msg="BasicDistributedModel: Network ouputs aren't equal!",
		)


	def test_basic_distributed_model_gradients(self):
		# Define the inputs.
		input_1 = torch.randn(BATCH_SIZE, NET_DIMS[0][0])
		input_2 = input_1[:]
		target_1 = torch.randn(BATCH_SIZE, NET_DIMS[-1][-1])
		target_2 = target_1[:]

		# Make the distributed model and get the gradients.
		model_2 = BasicDistributedModel(NET_DIMS, NUM_BATCHES, seed=True)
		grad_2 = model_2.forward_backward(input_2, target_2, return_grads=True)
		grad_2 = grad_2.detach().cpu().numpy()
		model_2.join()

		# Make the ground truth model and get the gradients.
		model_1 = NonDistributedModel(NET_DIMS, seed=True)
		grad_1 = model_1.forward_backward(input_1, target_1, return_grads=True)
		grad_1 = grad_1.detach().cpu().numpy()

		self.assertTrue( \
				np.allclose(grad_1, grad_2),
				msg="BasicDistributedModel: Network gradients aren't equal!",
		)



if __name__ == '__main__':
	unittest.main()



###
