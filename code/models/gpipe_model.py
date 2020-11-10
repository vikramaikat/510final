"""
A simple example, now with paralllelism and Gpipe-style microbatches.

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
		microbatch_losses, outputs = [], []

		# Perform forward passes.
		for batch in range(num_batches):
			# Receive data from our parent.
			backwards_flag, data = parent_conn.recv()
			# Return on None.
			if data is None:
				# If we're the final layer, plot the loss history.
				if final_layer:
					plot_loss(loss_values)
				# Then propogate the death signal.
				child_conn.send((None,None))
				return
			# Make a forward pass.
			output = net(data)
			# If we're just doing a forwards pass, just feed outputs to child.
			if not backwards_flag:
				child_conn.send((False, output.detach()))
				continue
			# If we're the last layer, calculate a loss.
			if final_layer:
				# Receive the targets.
				target = child_conn.recv()
				# Calculate a loss.
				loss = torch.mean(torch.pow(output - target, 2))
				microbatch_losses.append(loss)
			else:
				# Otherwise, pass the output to our child.
				child_conn.send((True, output.detach()))
				outputs.append(output)

		# If we're not doing a backwards flag, wait for more data.
		if not backwards_flag:
			continue

		# If we're the last layer, record and print loss.
		if final_layer:
			epoch_loss = sum(i.item() for i in microbatch_losses)
			epoch_loss /= num_batches
			loss_values.append(epoch_loss)
			if epoch % 100 == 0:
				print("epoch:", epoch ,"loss:", epoch_loss)

		# Zero gradients.
		optimizer.zero_grad()

		# Perform backward passes.
		for batch in range(num_batches-1,-1,-1):
			# Perform the backward pass.
			if final_layer:
				microbatch_losses[batch].backward()
			else:
				grads = child_conn.recv()
				outputs[batch].backward(gradient=grads)
			# Pass gradients back to the parent.
			parent_conn.send(net[0].bias.grad)
			# And zero out the NoOp layer.
			net[0].zero()
		# Update this module's parameters.
		optimizer.step()
		# And zero out the NoOp layer.
		net[0].zero()



class GpipeModel(DistributedModel):
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
		super(GpipeModel, self).__init__()
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
			# Release the process into the wild.
			p.start()


	def forward(self, x):
		"""Just the forward pass!"""
		# Send features to the first parition.
		self.pipes[0][0].send((False, x))
		# Get predictions from last partition.
		_, prediction = self.pipes[-1][1].recv()
		return prediction


	def forward_backward(self, x, y, wait_num=1):
		"""Both forward and backward passes."""
		# Send features to the first parition.
		self.pipes[0][0].send((True, x))
		# Send targets to the last parition.
		self.pipes[-1][1].send(y)
		for i in range(wait_num):
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
