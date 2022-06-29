import numpy as np
from .Base import BaseLayer


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        # super().__init__()
        self.tensor_shape = None
        self.testing_phase = False
        self.trainable = True
        # CHANNELS - is the number of channels in the input tensor
        self.channels = channels
        self.weights = None
        self.bias = None
        self.initialize(channels)

        self._optimizer = None

    def initialize(self, shape):
        self.weights = np.ones((shape))
        self.bias = np.zeros((shape))


    def forward(self, input_tensor):
        self.tensor_shape = input_tensor.shape
        # moving average provides better estimate of the true mean and variance compared to individual mini-batch
        # https://stackoverflow.com/questions/60460338/does-batchnormalization-use-moving-average-across-batches-or-only-per-batch-and

        # In training we need to calculate mini batch mean in order to normalize the batch
        # In the inference we just apply pre-calculated mini batch statistics
        if not self.testing_phase:
            # during training

            # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            # running_var = momentum * running_var + (1 - momentum) * sample_var
            eps = np.finfo(float).eps
            tensor = self.reformat(input_tensor)
            mean = np.mean(tensor, axis=0)
            var = np.var(tensor, axis=0)
            normalized_input_tensor = (input_tensor - mean)/np.sqrt(var + eps)
            y_bar = self.weights * normalized_input_tensor + self.bias
            return y_bar
        else:
            return

    def backward(self, error_tensor):
        return

    def reformat(self, tensor):

        if len(tensor.shape) > 2:

            tensor = tensor.reshape(self.tensor_shape[0], self.tensor_shape[1], np.prod(self.tensor_shape[2:]))
            tensor = np.swapaxes(tensor, 1, tensor.ndim-1)
            # reshape to B.M.N x H
            tensor = tensor.reshape(-1, self.tensor_shape[1])
            return tensor
        else:
            # if ka reverse will be here
            tensor = tensor.reshape(self.tensor_shape[0], -1, self.tensor_shape[1])
            tensor = np.swapaxes(tensor, 1, tensor.ndim-1)
            tensor = tensor.reshape(self.tensor_shape[0], self.tensor_shape[1], *(self.tensor_shape[2:]))
            return tensor
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.getter
    def optimizer(self):
        return self._optimizer
