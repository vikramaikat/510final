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
		labels.append(key)
		times.append(data['profiling'][key])
	print(labels)
	print(times)
	times = np.asarray(times)
	labels = np.asarray(labels)

	# From https://stackoverflow.com/questions/23577505/how-to-avoid-overlapping-of-labels-autopct-in-a-matplotlib-pie-chart
	percents = 100.*times/times.sum()
	patches, texts = plt.pie(times, startangle=90, shadow=True)
	legend = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(labels, percents)]
	sort_legend = True
	if sort_legend:
		patches, legend, dummy = zip(*sorted(zip(patches, legend, times), key=lambda labels: labels[2], reverse=True))
	plt.legend(patches, legend)
	plt.title(title)
	save_fn = 'plots/pie_plots/' + save_fn
	plt.savefig(save_fn)
	plt.close('all')


# Plot convergence for a given model and dataset with small (25), medium (50) and large (100) batch sizes
def plot_convergence_across_batches():
	return


# Plot time for a given model and dataset with small (25), medium (50) and large (100) batch sizes
def plot_time_across_batches():
	return


# Plot time for a all models and given dataset with a given batch sizes
def plot_time_across_models():
	return


if __name__ == '__main__':
	print("Running")
	print_pie_plot(2, 1, 1, 1, 25, 4, 200, 'test.pdf', 'Test Plot')
