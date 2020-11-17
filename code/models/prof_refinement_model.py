"""
Distributed model with a forward pass that sequentially refines the network
output. Also with some profiling.

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
FORWARD_FLAG = 1
TEST_FORWARD_FLAG = 2
PROFILING_CATEGORIES = [
	'zero_grad',
	'update_grad',
	'zero_no_op',
	'aux_forward',
	'forward',
	'backward',
	'blocked',
	'optim',
	'logging',
]



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
		output_net_mean = torch.nn.Linear(net_dims[-1], target_dim)
		# Define parameters.
		params = chain( \
				net.parameters(),
				output_net_mean.parameters(),
		)
	# Setup up profiling.
	prof = dict(zip(PROFILING_CATEGORIES, [0.0]*len(PROFILING_CATEGORIES)))
	# Make an optimizer.
	optimizer = torch.optim.Adam(params, lr=LR)
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
			backwards_flag, counter, data = parent_conn.recv()
			toc = time.perf_counter()
			prof['blocked'] += toc - tic

			# Return on None.
			if data is None:
				# If we're the final layer, plot the loss history.
				if final_layer:
					plot_loss(train_loss_values, train_loss_times, \
							test_loss_values, test_loss_times, test_loss_epochs)
				# Accumulate profiling times.
				incoming_prof = backwards_flag
				if incoming_prof is not None:
					assert type(incoming_prof) == type({})
					for key in prof:
						prof[key] += incoming_prof[key]
				# If we're in the last layer, also report the profiling results.
				if final_layer:
					print("Total profiling times:")
					print(prof)
					# Then propogate the None signal.
					child_conn.send((None,None,None))
				else:
					# Then propogate the None signal.
					child_conn.send((prof,None,None))
				return

			# If we've run out of time, keep propagating and move to the next
			# batch. Send the loss so that the final cell can keep an accurate
			# log.
			if counter < 0:
				assert backwards_flag
				# Empty target pipe.
				tic = time.perf_counter()
				_ = target_conn.recv()
				toc = time.perf_counter()
				prof['blocked'] += toc - tic
				# Send signal.
				child_conn.send((backwards_flag, counter, data))
				if final_layer:
					# Loss is stored in data if the forward pass didn't
					# complete.
					batch_losses.append(data)
				continue

			# Make a forward pass.
			tic = time.perf_counter()
			cell_output = net(data)
			toc = time.perf_counter()
			prof['forward'] += toc - tic

			if backwards_flag: # If we're going to do a backwards pass:

				# Zero gradients.
				tic = time.perf_counter()
				optimizer.zero_grad()
				toc = time.perf_counter()
				prof['zero_grad'] += toc - tic

				if final_layer:
					# If we're the last cell, calculate a loss.
					tic = time.perf_counter()
					target = target_conn.recv() # Receive the targets.
					toc = time.perf_counter()
					prof['blocked'] += toc - tic
					loss = torch.mean(torch.pow(cell_output - target, 2))
					batch_losses.append(loss.item())
					tic = time.perf_counter()
					loss.backward()
					toc = time.perf_counter()
					prof['backward'] += toc - tic
					# Pass gradients back.
					if counter > 0:
						parent_conn.send((counter-1, net[0].bias.grad))
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
					# Tell the main process that we're done.
					child_conn.send(None)
				else:

					if counter > 0:
						# If there's more computation to be done, pass the
						# output to our children.
						child_conn.send((True, counter-1, cell_output.detach()))

					# Then calculate the net output approximation.
					tic = time.perf_counter()
					est_target = output_net_mean(cell_output)
					toc = time.perf_counter()
					prof['aux_forward'] += toc - tic
					# Receive the targets.
					tic = time.perf_counter()
					target = target_conn.recv()
					toc = time.perf_counter()
					prof['blocked'] += toc - tic
					# Calculate a loss for this approximation.
					loss = torch.mean(torch.pow(est_target - target, 2))

					if counter == 0:
						# Otherwise, just propagate the loss forward for logging
						# purposes.
						child_conn.send((True, counter-1, loss.item()))

					# Perform a backward pass.
					tic = time.perf_counter()
					loss.backward(retain_graph=True)
					toc = time.perf_counter()
					prof['backward'] += toc - tic
					gradient_count = 1 # Count the number of backward calls.
					# Collect gradients from our children.
					while True:
						# Send gradients backward if they're needed.
						if layer_num > 0 and counter > 0:
							parent_conn.send((counter-1, net[0].bias.grad))
							tic = time.perf_counter()
							net[0].zero()
							toc = time.perf_counter()
							prof['zero_no_op'] += toc - tic
						# Break when we're not getting gradients again.
						if counter <= 1:
							break
						# Receive gradients.
						tic = time.perf_counter()
						counter, grads = child_conn.recv()
						toc = time.perf_counter()
						prof['blocked'] += toc - tic
						# Perform the backward pass.
						tic = time.perf_counter()
						cell_output.backward(gradient=grads, retain_graph=True)
						toc = time.perf_counter()
						prof['backward'] += toc - tic
						# Update the number of backward calls.
						gradient_count += 1
					# Normalize the gradients: equivalent to averaging the loss
					# values of all the approximations.
					if gradient_count > 1:
						tic = time.perf_counter()
						for param in net.parameters():
							param.grad /= gradient_count
						toc = time.perf_counter()
						prof['update_grad'] += toc - tic
					# Update this module's parameters.
					tic = time.perf_counter()
					optimizer.step()
					toc = time.perf_counter()
					prof['optim'] += toc - tic
					# Send a signal that we're done with the batch.
					if layer_num == 0:
						parent_conn.send(net[0].bias.grad)
					# And zero out the NoOp layer.
					tic = time.perf_counter()
					net[0].zero()
					toc = time.perf_counter()
					prof['zero_no_op'] += toc - tic
			else: # If we're just doing a forward pass:
				if final_layer and counter == TEST_FORWARD_FLAG:
					tic = time.perf_counter()
					target = target_conn.recv() # Receive the targets.
					toc = time.perf_counter()
					prof['blocked'] += toc - tic
					with torch.no_grad():
						loss = torch.mean(torch.pow(cell_output - target, 2))
					batch_losses.append(loss.item())
				# Just feed the activations to the child.
				child_conn.send((False, counter, cell_output.detach()))

		if not final_layer:
			continue

		if not backwards_flag and counter == FORWARD_FLAG:
			epoch -= 1
			continue

		tic = time.perf_counter()
		# Log the time and loss.
		epoch_loss = round(sum(batch_losses) / num_batches, 7)
		elapsed_time = round(time.perf_counter() - start_time, 5)
		if backwards_flag:
			train_loss_values.append(epoch_loss)
			train_loss_times.append(elapsed_time)
			# Print out a loss.
			if final_layer and backwards_flag and (epoch % 100 == 0):
				print("epoch:", epoch ,"loss:", epoch_loss, \
						"time:", elapsed_time)
		else:
			epoch -= 1 # Don't count this epoch.
			test_loss_values.append(epoch_loss)
			test_loss_times.append(elapsed_time)
			test_loss_epochs.append(epoch)
		toc = time.perf_counter()
		prof['logging'] += toc - tic



class ProfRefinementModel(DistributedModel):
	"""
	Network made up of distinct parameter partitions in different processes.

	Trained with standard backprop, but with a modified loss function.

	"""

	def __init__(self, net_dims, num_batches):
		"""
		Parameters
		----------
		net_dims : list of list of int
		num_batches : int
		"""
		super(ProfRefinementModel, self).__init__()
		assert len(net_dims) > 1
		self.net_dims = net_dims
		self.num_batches = num_batches
		# Make all the pipes.
		self.pipes = [mp.Pipe() for _ in range(len(net_dims)+1)]
		self.target_pipes = [mp.Pipe() for _ in range(len(net_dims))]
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
							net_dims[-1][-1],
					),
			)
			self.processes.append(p)
			p.start()


	def forward(self, x):
		"""Just the forward pass!"""
		# Send features to the first cell.
		self.pipes[0][0].send((False, FORWARD_FLAG, x))
		# Get predictions from last cell.
		_, _, prediction = self.pipes[-1][1].recv()
		return prediction


	def test_forward(self, x, y, wait=False):
		"""Forward pass without gradients, but log the loss."""
		# Send features to the first cell.
		self.pipes[0][0].send((False, TEST_FORWARD_FLAG, x))
		# Send targets to the last cell.
		self.target_pipes[-1][0].send(y)
		# Wait for something to come back.
		_ = self.pipes[-1][1].recv()


	def forward_backward(self, x, y, wait=False, counter=None):
		"""Both forward and backward passes."""
		if counter is None:
			counter = np.random.randint(2*len(self.net_dims)-1)
		assert counter >= 0 and counter <= 2 * len(self.net_dims) - 2
		# Send features to the first parition.
		self.pipes[0][0].send((True, counter, x))
		# Send targets to all the cells.
		for pipe in self.target_pipes:
			pipe[0].send(y)
		_ = self.pipes[0][0].recv() # Wait for gradients to come back.
		_ = self.pipes[-1][1].recv() # Wait for activations too.


	def join(self):
		"""Join all the processes."""
		# Send a kill signal.
		self.pipes[0][0].send((None, None, None))
		# Wait for it to come round.
		temp = self.pipes[-1][1].recv()
		assert temp == (None, None, None)
		# Join everything.
		for p in self.processes:
			p.join()



if __name__ == '__main__':
	pass


###
