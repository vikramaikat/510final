"""
Distributed model with a forward pass that sequentially refines the network
output.

"""
__date__ = "November 2020"


from itertools import chain
import multiprocessing as mp
import numpy as np
import torch
import time

from .distributed_model import DistributedModel
from .utils import make_dense_net, plot_loss, plot_performance

# torch.autograd.set_detect_anomaly(True) # useful for debugging

LR = 1e-3 # learning rate
AUX_LOSS_FACTOR = 0.5



def mp_target_func(net_dims, parent_conn, child_conn, target_conn, final_layer,\
	num_batches, layer_num, target_dim):
	"""
	Spawn a torch.nn.Module and wait for data.

	This is the meat of the parallelization.

	Parameters
	----------
	net_dims : list of int
		List of layer dimensions
	parent_conn : multiprocessing.connection.Connection
	child_conn : multiprocessing.connection.Connection
	target_conn : multiprocessing.connection.Connection
	final_layer : bool
	num_batches : int
	layer_num : int
	"""
	# Make a network.
	net = make_dense_net(net_dims, include_last_relu=(not final_layer))
	if final_layer:
		# Define parameters.
		params = net.parameters()
	else:
		# Make the network output approximation layers.
		output_net_mean = torch.nn.Linear(net_dims[-1][-1], target_dim)
		# Define parameters.
		params = chain( \
				net.parameters(),
				output_net_mean.parameters(),
		)
	# Make an optimizer.
	optimizer = torch.optim.Adam(params, lr=LR)
	# Enter main loop.
	epoch = 0
	loss_values = []
	start_time = time.perf_counter()
	while True:
		epoch += 1
		batch_losses = []

		for batch in range(num_batches):

			# Receive data from our parent.
			backwards_flag, counter, data = parent_conn.recv()

			# Return on None.
			if data is None:
				# If we're the final layer, plot the loss history.
				if final_layer:
					plot_loss(loss_values)
				# Then propogate the None signal.
				child_conn.send((None,None,None))
				return

			# If we've run out of time, keep propagating and move to the next
			# batch.
			if counter < 0:
				# Empty pipes.
				_ = target_conn.recv()
				# Send signal.
				child_conn.send(backwards_flag, counter, True)
				continue

			# Zero gradients: MOVE THIS!
			optimizer.zero_grad()

			# Make a forward pass.
			cell_output = net(data)

			if backwards_flag: # If we're going to do a backwards pass:
				if final_layer:
					# If we're the last cell, calculate a loss.
					target = target_conn.recv() # Receive the targets.
					loss = torch.mean(torch.pow(cell_output - target, 2))
					batch_losses.append(loss.item())
					loss.backward()
					# Pass gradients back.
					if counter > 0:
						parent_conn.send(net[0].bias.grad)
					# Update this module's parameters.
					optimizer.step()
					# And zero out the NoOp layer.
					net[0].zero()
				else:
					# Pass the output to our children.
					child_conn.send((True, counter-1, cell_output.detach()))
					# Then calculate the net output approximation.
					est_target = output_net_mean(cell_output)
					# Calculate a loss for this approximation.
					target = target_conn.recv() # Receive the targets.
					loss = torch.mean(torch.pow(est_target - target, 2))
					loss = loss * AUX_LOSS_FACTOR
					loss.backward()
					while True: # Some tricky issues here.
						# Send gradients backward if they're needed.
						if counter > 0 and counter <= layer_num + 1:
							parent_conn.send(net[0].bias.grad)
						# Zero gradients if we're going to get better ones.
						# NOTE: HERE!

						grads = child_conn.recv()
						# Perform the backward pass.
						cell_output.backward(gradient=grads)
				# Pass gradients back.
				parent_conn.send(net[0].bias.grad)
				# Update this module's parameters.
				optimizer.step()
				# And zero out the NoOp layer.
				net[0].zero()
			else: # If we're just doing a forwards pass:
				# Just feed the activations to the child.
				child_conn.send((False, None, cell_output.detach()))

		# Print out a loss.
		if final_layer and backwards_flag and (epoch % 100 == 0):
			epoch_loss = sum(batch_losses) / num_batches
			loss_values.append(epoch_loss)
			print("epoch:", epoch ,"loss:", epoch_loss)
			print("time:", time.perf_counter() - start_time)



class RefinementModel(DistributedModel):
	"""
	Network made up of distinct parameter partitions in different processes.

	Trained with standard backprop.

	"""

	def __init__(self, net_dims, num_batches):
		"""
		Parameters
		----------
		net_dims : ...
		num_batches : int
		"""
		super(RefinementModel, self).__init__()
		self.num_batches = num_batches
		# Make all the pipes.
		self.pipes = [mp.Pipe() for _ in range(len(net_dims)+1)]
		self.target_pipes = [mp.Pipe() for _, in range(len(net_dims))]
		# Make each network.
		self.processes = []
		for i, sub_net_dims in enumerate(net_dims):
			conn_1 = self.pipes[i][1]
			conn_2 = self.pipes[i+1][0]
			target_conn = self.target_pipes[i][1]
			final_layer = (i == len(net_dims)-1)
			p = mp.Process( \
					target=mp_target_func,
					args=( \
							sub_net_dims,
							conn_1,
							conn_2,
							target_conn,
							final_layer,
							self.num_batches,
							i,
					),
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


	def forward_backward(self, x, y, wait_num=None):
		"""Both forward and backward passes."""
		# Send features to the first parition.
		self.pipes[0][0].send((True, x))
		# Send targets to the last parition.
		self.pipes[-1][1].send(y)
		_ = self.pipes[0][0].recv() # Wait for gradients to come back.


	def join(self):
		"""Join all the processes."""
		# Send a kill signal.
		self.pipes[0][0].send((None,None))
		# Wait for it to come round.
		temp = self.pipes[-1][1].recv()
		assert temp == (None, None)
		# Join everything.
		for p in self.processes:
			p.join()



if __name__ == '__main__':
	pass


###
