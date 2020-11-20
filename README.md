# 510final

### File Summaries

#### models/utils.py
This file is used to store functions that are useful across models. Here we have the basic structure
for the neural net, the loss plot and the performance plot functions.

#### models/no_op.py
Define a No-Op Pytorch Module that will be useful for caching gradients. 
Adds a 0 bias term to the input but because the bias is a tensor it then stores the gradients. 

#### models/gpipe_model.py
Builds upon the distributed model by using the GPipe microbatching technique. 
#### models/distributed_model.py
Defines the abstract class for all distributed models we are  using. Requires forward_backward and join methods. 

#### models/basic_distributed_model.py
This model uses communication pipes from the Multiprocessing library in python. FOr each cell / subnetwork you make a different
MP process and give it connections (pipes) to the next layer and from the previous layer to send activations and gradient.

#### data/xor_data.py
Defines the XOR_data set. 

#### data/mnist_data.py
Defines the Mnist_data set. Goal of this dataset is to predict the bottom half of the image given the top half. 

#### models/prof_*
All of these models are identical to their non profiling counterpart, except these log time spent on different tasks
such as blocking, forward passes, backward passes and zero-grad. 

#### refinement_model.py
Refinement mode, still feedforward, split into chunks, but each subnetwork is a cell and each has an extra network attached onto it. 
Follows the idea of “If you had to stop right now and guess what the output was, what was it going to be”. 

#### read_np_files.py
This script is used to read the logging outputs and plot their results from the profiling models. 

#### Sources
All code outside of the plotting scripts are original. The code for plotting a pieplot with a legend was 
taken from https://stackoverflow.com/questions/23577505/how-to-avoid-overlapping-of-labels-autopct-in-a-matplotlib-pie-chart. 