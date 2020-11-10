"""
A basic distributed model.


"""
__date__ = "October - November 2020"


import multiprocessing as mp
import numpy as np
import torch

from .distributed_model import DistributedModel
from .utils import make_dense_net, plot_loss

# torch.autograd.set_detect_anomaly(True) # useful for debugging

LR = 1e-3 # learning rate



def mp_target_func(net_dims, parent_conn, child_conn, final_layer, num_batches):
	"""
	Spawn a torch.nn.Module and wait for data.

	This is the meat of the parallelization.

	Parameters
	----------
	net_dims : list of int
		List of layer dimensions
	parent_conn : multiprocessing.connection.Connection
	child_conn : multiprocessing.connection.Connection
	final_layer : bool
	num_batches : int
	"""
	# Make a network.
	net = make_dense_net(net_dims, include_last_relu=(not final_layer))
	# Make an optimizer.
	optimizer = torch.optim.Adam(net.parameters(), lr=LR)
	# Enter main loop.
	epoch = 0
	loss_values = []
	while True:
		epoch += 1
		batch_losses = []

		for batch in range(num_batches):
			# Receive data from our parent.
			backwards_flag, data = parent_conn.recv()
			# Return on None.
			if data is None:
				# If we're the final layer, plot the loss history.
				if final_layer:
					plot_loss(loss_values)
				# Then propogate the None signal.
				child_conn.send((None,None))
				return
			# Zero gradients.
			optimizer.zero_grad()
			# Make a forward pass.
			output = net(data)
			if backwards_flag: # If we're going to do a backwards pass:
				if final_layer:
					# If we're the last Module, calculate a loss.
					target = child_conn.recv() # Receive the targets.
					loss = torch.mean(torch.pow(output - target, 2))
					batch_losses.append(loss.item())
					loss.backward()
				else:
					# Otherwise, pass the output to our children.
					child_conn.send((True, output.detach()))
					grads = child_conn.recv()
					# Perform the backward pass.
					output.backward(gradient=grads)
				# Pass gradients back.
				parent_conn.send(net[0].bias.grad)
				# Update this module's parameters.
				optimizer.step()
				# And zero out the NoOp layer.
				net[0].zero()
			else: # If we're just doing a forwards pass:
				# Just feed the activations to the child.
				child_conn.send((False, output.detach()))

		# Print out a loss.
		if final_layer and backwards_flag and (epoch % 100 == 0):
			epoch_loss = sum(batch_losses) / num_batches
			loss_values.append(epoch_loss)
			print("epoch:", epoch ,"loss:", epoch_loss)



class BasicDistributedModel(DistributedModel):
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
		super(BasicDistributedModel, self).__init__()
		self.num_batches = num_batches
		# Make all the pipes.
		self.pipes = [mp.Pipe() for _ in range(len(net_dims)+1)]
		# Make each network.
		self.processes = []
		for i, sub_net_dims in enumerate(net_dims):
			conn_1 = self.pipes[i][1]
			conn_2 = self.pipes[i+1][0]
			final_layer = (i == len(net_dims)-1)
			p = mp.Process( \
					target=mp_target_func,
					args=( \
							sub_net_dims,
							conn_1,
							conn_2,
							final_layer,
							self.num_batches,
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
