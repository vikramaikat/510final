"""
A basic distributed model with some profiling.

This is separate from basic_distributed_model so we can tell how much of a
performance hit the profiling is.
"""
__date__ = "October - November 2020"


import multiprocessing as mp
import numpy as np
import os
import torch
import time

from .distributed_model import DistributedModel
from .utils import make_dense_net, plot_loss

# torch.autograd.set_detect_anomaly(True) # useful for debugging

LR = 1e-3  # learning rate
TRAIN_FLAG = 0
TEST_FLAG = 1
FORWARD_FLAG = 2
PRINT_FREQ = 10
PROFILING_CATEGORIES = [
	'zero_grad',
	'zero_no_op',
	'forward',
	'backward',
	'blocked',
	'optim',
	'logging',
	'wall_clock',
]



def mp_target_func(net_dims, parent_conn, child_conn, final_layer, num_batches,
	save_fn):
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
	save_fn : None or str
	"""
	# Make a network.
	net = make_dense_net(net_dims, include_last_relu=(not final_layer))
	# Make an optimizer.
	optimizer = torch.optim.Adam(net.parameters(), lr=LR)
	# Setup up profiling.
	prof = dict(zip(PROFILING_CATEGORIES, [0.0]*len(PROFILING_CATEGORIES)))
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
			tic = time.perf_counter()
			epoch_flag, data = parent_conn.recv()
			toc = time.perf_counter()
			prof['blocked'] += toc - tic

			# Return on None.
			if data is None:
				# If we're the final layer, plot the loss history.
				if final_layer:
					plot_loss(train_loss_values, train_loss_times, \
							test_loss_values, test_loss_times, test_loss_epochs)
				# Accumulate profiling times.
				incoming_prof = epoch_flag
				if incoming_prof is not None:
					assert type(incoming_prof) == type({})
					for key in prof:
						prof[key] += incoming_prof[key]
				else:
					# Add the wall-clock time.
					prof['wall_clock'] = time.perf_counter() - start_time
				# If we're in the last layer, also report the profiling results.
				if final_layer:
					print("Total profiling times:")
					print(prof)
					if save_fn is not None:
						np.save( \
							save_fn,
							{
								'train_loss': train_loss_values,
								'train_time': train_loss_times,
								'test_loss': test_loss_values,
								'test_time': test_loss_times,
								'test_epochs': test_loss_epochs,
								'profiling': prof,
							}
						)
					# Then propogate the None signal.
					child_conn.send((None, None))
				else:
					# Then propogate the None signal.
					child_conn.send((prof, None))
				return

			assert epoch_flag in [TRAIN_FLAG, TEST_FLAG, FORWARD_FLAG, None]

			# Zero gradients.
			tic = time.perf_counter()
			optimizer.zero_grad()
			toc = time.perf_counter()
			prof['zero_grad'] += toc - tic

			# Make a forward pass.
			tic = time.perf_counter()
			output = net(data)
			toc = time.perf_counter()
			prof['forward'] += toc - tic

			# If we're going to do a backwards pass:
			if epoch_flag == TRAIN_FLAG:
				if final_layer:
					# If we're the last Module, calculate a loss.
					tic = time.perf_counter()
					target = child_conn.recv() # Receive the targets.
					toc = time.perf_counter()
					prof['blocked'] += toc - tic

					loss = torch.mean(torch.pow(output - target, 2))
					batch_losses.append(loss.item())
					tic = time.perf_counter()
					loss.backward()
					toc = time.perf_counter()
					prof['backward'] += toc - tic
				else:
					# Otherwise, pass the output to our children.
					child_conn.send((TRAIN_FLAG, output.detach()))
					tic = time.perf_counter()
					grads = child_conn.recv()
					toc = time.perf_counter()
					prof['blocked'] += toc - tic

					# Perform the backward pass.
					tic = time.perf_counter()
					output.backward(gradient=grads)
					toc = time.perf_counter()
					prof['backward'] += toc - tic
				# Pass gradients back.
				parent_conn.send(net[0].bias.grad)
				# Update this module's parameters.
				tic = time.perf_counter()
				optimizer.step()
				toc = time.perf_counter()
				prof['optim'] += toc - tic
				# And zero out the NoOp layer.
				tic = time.perf_counter()
				net[0].zero()
				toc = time.perf_counter()
				prof['zero_no_op'] += toc - tic
			elif epoch_flag == FORWARD_FLAG or \
					(epoch_flag == TEST_FLAG and not final_layer):
				# Just feed the activations to the child.
				child_conn.send((epoch_flag, output.detach()))
			elif final_layer and epoch_flag == TEST_FLAG:
				tic = time.perf_counter()
				target = child_conn.recv() # Receive the targets.
				toc = time.perf_counter()
				prof['blocked'] += toc - tic
				with torch.no_grad():
					loss = torch.mean(torch.pow(output - target, 2))
				batch_losses.append(loss.item())
				# Send something to the main process to signal we're done.
				child_conn.send(None)

		if not final_layer:
			continue

		tic = time.perf_counter()
		if epoch_flag != TRAIN_FLAG:
			epoch -= 1
		# Log the time and loss.
		epoch_loss = round(sum(batch_losses) / num_batches, 7)
		elapsed_time = round(time.perf_counter() - start_time, 5)
		if epoch_flag == TRAIN_FLAG:
			train_loss_values.append(epoch_loss)
			train_loss_times.append(elapsed_time)
			# Print out a loss.
			if epoch % PRINT_FREQ == 0:
				print("epoch:", epoch ,"loss:", epoch_loss, \
						"time:", elapsed_time)
		elif epoch_flag == TEST_FLAG:
			test_loss_values.append(epoch_loss)
			test_loss_times.append(elapsed_time)
			test_loss_epochs.append(epoch)
		toc = time.perf_counter()
		prof['logging'] += toc - tic



class ProfBasicDistributedModel(DistributedModel):
	"""
	Network made up of distinct parameter partitions in different processes.

	Trained with standard backprop.

	"""

	def __init__(self, net_dims, num_batches, cpu_affinity=False, save_fn=None):
		"""
		Parameters
		----------
		net_dims : list of list of int
		num_batches : int
		cpu_affinity : bool, optional
		save_fn : None or str, optional
		"""
		super(ProfBasicDistributedModel, self).__init__()
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
							save_fn,
					),
			)
			self.processes.append(p)
		# Pin the processes to specific CPUs.
		if cpu_affinity:
			# First pin ourselves down to a CPU.
			cpu_count = os.cpu_count()
			os.system("taskset -p -c %d %d" % (-1 % cpu_count, os.getpid()))
		# Release the processes into the wild.
		for p in self.processes:
			p.start()
		# Then pin everyone else down in a round-robin.
		if cpu_affinity:
			for i, p in enumerate(self.processes):
				os.system("taskset -p -c %d %d" % (i % cpu_count, p.pid))


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
