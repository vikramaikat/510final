"""
A basic distributed model.

"""
__date__ = "October - November 2020"


import multiprocessing as mp
import numpy as np
import torch
import time

from .distributed_model import DistributedModel
from .utils import make_dense_net, plot_loss

# torch.autograd.set_detect_anomaly(True) # useful for debugging

LR = 1e-3  # learning rate
TRAIN_FLAG = 0
TEST_FLAG = 1
FORWARD_FLAG = 2


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
	train_loss_values, train_loss_times = [], []
	test_loss_values, test_loss_times, test_loss_epochs = [], [], []
	start_time = time.perf_counter()
	while True:
		epoch += 1
		batch_losses = []

		for batch in range(num_batches):

			# Receive data from our parent.
			epoch_flag, data = parent_conn.recv()
			assert epoch_flag in [TRAIN_FLAG, TEST_FLAG, FORWARD_FLAG, None]

			# Return on None.
			if data is None:
				# If we're the final layer, plot the loss history.
				if final_layer:
					plot_loss(train_loss_values, train_loss_times, \
							test_loss_values, test_loss_times, test_loss_epochs)
				# Then propogate the None signal.
				child_conn.send((None,None))
				return

			# Zero gradients.
			optimizer.zero_grad()

			# Make a forward pass.
			output = net(data)

			# If we're going to do a backwards pass:
			if epoch_flag == TRAIN_FLAG:
				if final_layer:
					# If we're the last Module, calculate a loss.
					target = child_conn.recv() # Receive the targets.
					loss = torch.mean(torch.pow(output - target, 2))
					batch_losses.append(loss.item())
					loss.backward()
				else:
					# Otherwise, pass the output to our children.
					child_conn.send((TRAIN_FLAG, output.detach()))
					grads = child_conn.recv()
					# Perform the backward pass.
					output.backward(gradient=grads)
				# Pass gradients back.
				parent_conn.send(net[0].bias.grad)
				# Update this module's parameters.
				optimizer.step()
				# And zero out the NoOp layer.
				net[0].zero()
			elif epoch_flag == FORWARD_FLAG or \
					(epoch_flag == TEST_FLAG and not final_layer):
				# Just feed the activations to the child.
				child_conn.send((epoch_flag, output.detach()))
			elif final_layer and epoch_flag == TEST_FLAG:
				target = child_conn.recv() # Receive the targets.
				with torch.no_grad():
					loss = torch.mean(torch.pow(output - target, 2))
				batch_losses.append(loss.item())
				# Send something to the main process to signal we're done.
				child_conn.send(None)

		if not final_layer:
			continue

		if epoch_flag != TRAIN_FLAG:
			epoch -= 1

		# Log the time and loss.
		epoch_loss = round(sum(batch_losses) / num_batches, 7)
		elapsed_time = round(time.perf_counter() - start_time, 5)

		if epoch_flag == TRAIN_FLAG:
			train_loss_values.append(epoch_loss)
			train_loss_times.append(elapsed_time)
			# Print out a loss.
			if epoch % 100 == 0:
				print("epoch:", epoch ,"loss:", epoch_loss, \
						"time:", elapsed_time)
		elif epoch_flag == TEST_FLAG:
			test_loss_values.append(epoch_loss)
			test_loss_times.append(elapsed_time)
			test_loss_epochs.append(epoch)



class BasicDistributedModel(DistributedModel):
	"""
	Network made up of distinct parameter partitions in different processes.

	Trained with standard backprop.

	"""

	def __init__(self, net_dims, num_batches):
		"""
		Parameters
		----------
		net_dims : list of list of int
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
		# Send features to the first cell.
		self.pipes[0][0].send((FORWARD_FLAG, x))
		# Get predictions from last cell.
		_, prediction = self.pipes[-1][1].recv()
		return prediction


	def test_forward(self, x, y, wait=False):
		"""Forward pass without gradients, but log the loss."""
		# Send features to the first cell.
		self.pipes[0][0].send((TEST_FLAG, x))
		# Send targets to the last cell.
		self.pipes[-1][1].send(y)
		# Wait for a response from the last cell.
		_ = self.pipes[-1][1].recv()


	def forward_backward(self, x, y, wait=False):
		"""Both forward and backward passes."""
		# Send features to the first cell.
		self.pipes[0][0].send((TRAIN_FLAG, x))
		# Send targets to the last cell.
		self.pipes[-1][1].send(y)
		_ = self.pipes[0][0].recv() # Wait for gradients to come back.
		return _


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
