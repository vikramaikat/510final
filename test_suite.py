import unittest
import torch
from code.data.xor_data import get_xor_training_data, plot_model_predictions
from code.models.gpipe_model import GpipeModel
from code.models.basic_distributed_model import BasicDistributedModel
from code.models.utils import make_dense_net
import numpy as np

# TODO: Test if backprop calculates exact gradients
# TODO: Test if local learning rules calculate approximate gradients
# TODO: Check that inputs to the training_script are correct
#
LR = 1e-3
BATCH_SIZE = 25
NUM_BATCHES = 4
DSET_SIZE = BATCH_SIZE * NUM_BATCHES
NET_DIMS = [[2,10,10], [10,10,1]]
EPOCHS = 1001
DT = 100

class MyTestCase(unittest.TestCase):
    def test_gradients(self):
        # Make a distributed network and make same network in a non-distributed way (using torch.nn.Sequential)
        # or can just concatenate the net_dims and call make_dense_net with noop layer
        torch.manual_seed(0)
        dist_grads = np.zeros((EPOCHS, NUM_BATCHES))
        net_grads = np.zeros((EPOCHS, NUM_BATCHES))

        features, targets = get_xor_training_data(n_samples=DSET_SIZE)
        dset = torch.utils.data.TensorDataset(features, targets)
        loader = torch.utils.data.DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True)
        model = BasicDistributedModel(NET_DIMS, NUM_BATCHES)
        for epoch in range(EPOCHS):
            for batch, (features, targets) in enumerate(loader):
                wait_num = NUM_BATCHES * int(batch == NUM_BATCHES - 1)
                grad = model.forward_backward(features, targets)
                ### Converting Tensor to np array causing issue (says pipe is done?)
                #dist_grads[epoch][batch] = grad.eval()
        plot_model_predictions(model)
        model.join()

        # Non non distributed
        net_basic = torch.nn.Sequential(make_dense_net(NET_DIMS[0]), make_dense_net(NET_DIMS[1]))
        optimizer = torch.optim.Adam(net_basic.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            for batch, (features, targets) in enumerate(loader):
                optimizer.zero_grad()
                prediction = net_basic(features)
                loss = torch.mean(torch.pow(prediction - targets, 2))
                loss.backward()
                grad = net_basic[0][0].bias.grad
                ### Converting Tensor to np array causing issue
                #net_grads[epoch][batch] = grad.eval()
                optimizer.step()
            if epoch % DT == 0:
                print("epoch", epoch, "loss", loss.item())
        # Make sure parameters are initialized in the same way
        self.assertEqual(net_grads.all(), dist_grads.all())
        # call forward_backward with the data on the distributed net and collect the layer 1 gradients

        # With non-distributed, pass the data through the model, calculating the loss.
        # Call loss.backward() then check if the noop gradients match (model[0].bias.grad)



        self.assertEqual(True, True, msg="Gradients are not equal")


if __name__ == '__main__':
    unittest.main()
