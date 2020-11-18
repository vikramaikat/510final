"""
Test suite for refinement model.

Contains
--------
* test_refinement_model_activations

"""
__date__ = "November 2020"


import numpy as np
import torch
import unittest

from code.models.refinement_model import RefinementModel
from code.models.non_distributed_model import NonDistributedModel


NUM_BATCHES = 4
BATCH_SIZE = 25
NET_DIMS = [[2,10,10], [10,10,1]]



class RefinementModelTestCases(unittest.TestCase):

	def test_refinement_model_activations(self):
		# Define the inputs.
		input_1 = torch.randn(BATCH_SIZE, NET_DIMS[0][0])
		input_2 = input_1[:]

		# Make the ground truth model and get activations.
		model_1 = NonDistributedModel(NET_DIMS, seed=True)
		with torch.no_grad():
			output_1 = model_1(input_1).cpu().numpy()

		# Make the distributed model and get activations.
		model_2 = RefinementModel(NET_DIMS, NUM_BATCHES, seed=True)
		with torch.no_grad():
			output_2 = model_2(input_2).cpu().numpy()
		model_2.join()

		self.assertTrue( \
				np.allclose(output_1, output_2),
				msg="RefinementModel: Network ouputs aren't equal!",
		)



if __name__ == '__main__':
	unittest.main()



###
