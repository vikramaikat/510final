import numpy as np
import os
import matplotlib.pyplot as plt


# For each of the three models: Run profiling and get the breakdown
# Do this with both the large and small model, with each having small (25), medium (50) and large (100) batch sizes

# python training_script.py [model_type] [dataset_type] [cpu_affinity] [model_size] [batch_size] [n_batch] [epochs] [save_fn]
def print_pie_plot(model_type, dataset_type, cpu_affinity, model_size, batch_size, n_batch, epochs, save_fn, title):
	exec_statement = 'python training_script.py ' + str(model_type) + ' ' + str(dataset_type) + ' ' + str(cpu_affinity) + \
					' ' + str(model_size) + ' ' + str(batch_size) + ' ' + str(n_batch) + ' ' + str(epochs) + ' ' + 'results.npy'
	os.system(exec_statement)

	data = np.load('results.npy', allow_pickle=True).item()

	labels = []
	times = []
	for key in data['profiling'].keys():
		if key != 'wall_clock':
			labels.append(key)
			times.append(data['profiling'][key])
	print(labels)
	print(times)
	times = np.asarray(times)
	labels = np.asarray(labels)
	colours = {'blocked': 'C0', 'optim': 'C1', 'backward': 'C2', 'forward': 'C3', 'zero_grad': 'C4', 'logging': 'C5', 'zero_no_op': 'C6', 'update_grad' : 'C7', 'aux_forward': 'C8'}
	#colors = {'logging': 'purple', 'brown', 'green', 'red', 'blue', 'orange', 'pink']
	# From https://stackoverflow.com/questions/23577505/how-to-avoid-overlapping-of-labels-autopct-in-a-matplotlib-pie-chart
	percents = 100.*times/times.sum()
	patches, texts = plt.pie(times, startangle=90, shadow=True, colors=[colours[key] for key in labels])
	legend = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(labels, percents)]
	sort_legend = True
	if sort_legend:
		patches, legend, dummy = zip(*sorted(zip(patches, legend, times), key=lambda labels: labels[2], reverse=True))
	plt.legend(patches, legend, loc='lower left')
	plt.title(title)
	save_fn = 'plots/pie_plots/' + save_fn
	plt.savefig(save_fn)
	plt.close('all')


if __name__ == '__main__':
	print("Running")
	print_pie_plot(2, 1, 1, 1, 25, 4, 200, 'pie_basic2.pdf', 'Basic Model Profiling')
