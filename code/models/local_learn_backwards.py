"""
Distributed model with local learning for just the backwards pass.

NOTE: Not at all finished!
"""
__date__ = "November 2020"


import multiprocessing as mp
import numpy as np
import torch

from .distributed_model import DistributedModel
from .utils import make_dense_net, plot_loss

# torch.autograd.set_detect_anomaly(True) # useful for debugging

LR = 1e-3 # learning rate
REC_LR = 1e-3 # recognition model learning rate
EST_GRAD_P = 0.5 # est_grad := P (new_estimate) + (1-P) (est_grad)



def mp_target_func(net_dims, parent_conn, child_conn, rec_conn, first_layer, \
	final_layer, num_batches, feature_dim, target_dim):
	"""
	Spawn a torch.nn.Module and wait for data.

	This is the meat of the parallelization.

	Here's the plan for each batch:

	1) Perform the following tasks:
		a) Get activations from our parent and perform a forward pass, passing
			activations to children.
		b) Get features and targets and send through the recognition model to
			get estimated gradients.
		c) Perform local updates on gradients, communicating with neighbors in
			the computational graph.

		with precedence given to (a) and (c) happening only after (b).

	2) Terminate receiving i) the true gradients or ii) a stop signal. Both are
	   guaranteed to happen after 1a and 1b.

	3) Then perform the following tasks in order:
		a) Perform a backwards pass with current gradient estimates. Take a
		   gradient step.
		b) Calculate a loss for the recognition model. Perform a backwards pass
		   and take a gradient step.


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
	# Make the main optimizer.
	optimizer = torch.optim.Adam(net.parameters(), lr=LR)
	# Make a recognition model.
	rec_net = torch.nn.Linear(feature_dim+target_dim,net_dims[-1][-1])
	# Make the recognition model optimizer.
	rec_optimizer = torch.optim.Adam(rec_net.parameters(), lr=REC_LR)
	# Enter main loop.
	epoch = 0
	loss_values = []
	time_values = []
	start_time = time.perf_counter()
	while True:
		epoch += 1
		batch_losses = []

		for batch in range(num_batches):

			# Step 1.
			model_forward_flag, rec_forward_flag = False, False
			while True:

				# Receive activations from our parent.
				if not model_forward_flag and parent_conn.poll():
					backwards_flag, data = parent_conn.recv()
					# Return on None.
					if data is None:
						# If we're the final layer, plot the loss history.
						if final_layer:
							plot_loss(loss_values)
						# Then propogate the None signal.
						child_conn.send((None,None))
						return
					# Make a forward pass through the model.
					output = net(data)
					# Note that we've made a forward pass on the model.
					model_forward_flag = True

				# Receive features and targets from the master.
				if not rec_forward_flag and rec_conn.poll():
					features, targets = rec_conn.recv()
					rec_input = torch.cat([features,targets], dim=-1)
					# Estimate gradients.
					initial_est_grad = rec_net(rec_input)
					est_grad = initial_est_grad.detach()
					# Note we've made a forward pass on the recognition model.
					rec_forward_flag = True

				# Perform local updates on gradients.
				if rec_forward_flag:
					# Send my gradient beliefs to the parent thread.
					optimizer.zero_grad()
					net[0].zero()
					output.backward(gradient=est_grad)
					parent_conn.send(net[0].bias.grad) # Send a flag here too?

					# Update my gradient beliefs.
					if child_conn.poll():
						_, est_grad_update = child_conn.recv()
						est_grad *= 1.0 - EST_GRAD_P
						est_grad += EST_GRAD_P * est_grad_update

			# Make a backward pass.
			if backwards_flag: # If we're going to do a backwards pass:
				if final_layer:
					# If we're the last Module, calculate a loss.
					target = child_conn.recv() # Receive the targets.
					loss = torch.mean(torch.pow(output - target, 2)) # MSE
					batch_losses.append(loss.item())
					# Zero gradients.
					optimizer.zero_grad()
					# Perform the backward pass.
					loss.backward()
				else:
					# Otherwise, pass the output to our children.
					child_conn.send((True, output.detach()))
					grads = child_conn.recv()
					# Zero gradients.
					optimizer.zero_grad()
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
			time_values.append(time.perf_counter() - start_time)
			elapsed_time = time.perf_counter() - start_time
			print("epoch:", epoch ,"loss:", epoch_loss, "time:", elapsed_time)



class LocalBackwardsModel(DistributedModel):
	"""
	Network made up of distinct parameter partitions in different processes.

	Trained with local learning backprop.
	"""

	def __init__(self, net_dims, num_batches):
		"""
		Parameters
		----------
		net_dims : ...
		num_batches : int
		"""
		super(LocalBackwardsModel, self).__init__()
		assert len(net_dims) > 1
		self.num_batches = num_batches
		# Make all the pipes:
		# local connection pipes: forms a loop
		self.pipes = [mp.Pipe() for _ in range(len(net_dims)+1)]
		# recognition model pipes: one to all
		self.rec_pipes [mp.Pipe() for _ in range(len(net_dims))]
		# Make each network.
		self.processes = []
		for i, sub_net_dims in enumerate(net_dims):
			parent_conn = self.pipes[i][1] # local connection parent
			child_conn = self.pipes[i+1][0] # local connection child
			rec_conn = self.rec_pipes[i][1] # recognition model connection
			final_layer = (i == len(net_dims)-1)
			first_layer = (i == 0)
			p = mp.Process( \
					target=mp_target_func,
					args=( \
							sub_net_dims,
							parent_conn,
							child_conn,
							rec_conn,
							first_layer,
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
