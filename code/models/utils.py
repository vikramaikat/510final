"""
Useful functions for distributed models.

"""
__date__ = "November 2020"


import matplotlib.pyplot as plt
import numpy as np
import torch

from .no_op import NoOp



def make_dense_net(layer_dims, include_no_op=True, include_last_relu=True):
	"""
	Make a network with ReLUs and dense layers.

	Parameters
	----------
	layer_dims : list of int
		List of layer dimensions
	include_no_op : bool, optional
	include_last_relu : bool, optional
	"""
	assert len(layer_dims) >= 2
	layer_list = []
	if include_no_op:
		layer_list.append(NoOp(layer_dims[0]))
	for i in range(len(layer_dims)-1):
		layer_list.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1]))
		if include_last_relu or i != len(layer_dims)-2:
			layer_list.append(torch.nn.ReLU())
	return torch.nn.Sequential(*layer_list).to(torch.float)


def plot_loss(loss_values, img_fn='loss.pdf'):
	"""Make a loss plot: log MSE by epoch"""
	_, ax = plt.subplots(figsize=(3.5,3))
	plt.plot(np.log(loss_values), alpha=0.7, lw=0.7)
	plt.title('Loss History')
	plt.ylabel('log MSE')
	plt.xlabel('Epoch')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	plt.tight_layout()
	plt.savefig(img_fn)
	plt.close('all')


def plot_performance(loss_values, time_values, data_samples, img1_fn='ovd.pdf', img2_fn='dvt.pdf', img3_fn='convergence.pdf'):
	"""Make a loss plot: log MSE by epoch"""
	_, ax = plt.subplots(figsize=(3.5,3))
	plt.plot(data_samples, np.log(loss_values), alpha=0.7, lw=0.7)
	plt.title('Loss vs Data Samples')
	plt.ylabel('log MSE')
	plt.xlabel('Samples Operated Upon')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	plt.tight_layout()
	plt.savefig(img1_fn)
	plt.close('all')

	"""Make a loss plot: log MSE by epoch"""
	_, ax = plt.subplots(figsize=(3.5, 3))
	plt.plot(time_values, data_samples, alpha=0.7, lw=0.7)
	plt.title('Data Samples over Time')
	plt.ylabel('Samples Operated Upon')
	plt.xlabel('Time (s)')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	plt.tight_layout()
	plt.savefig(img2_fn)
	plt.close('all')

	"""Make a loss plot: log MSE by epoch"""
	_, ax = plt.subplots(figsize=(3.5, 3))
	plt.plot(time_values, np.log(loss_values), alpha=0.7, lw=0.7)
	plt.title('Convergence')
	plt.ylabel('log MSE')
	plt.xlabel('Time (s)')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	plt.tight_layout()
	plt.savefig(img3_fn)
	plt.close('all')



if __name__ == '__main__':
	pass



###
