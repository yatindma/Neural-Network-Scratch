import numpy as np
from .Base import BaseLayer
from .Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        # super().__init__()
        self._gradient_bias = None
        self._gradient_weights = None
        self._bias_optimizer = None
        self.g_b = None
        self.g_w = None
        self.variance = 1
        self.mean = 0
        self.tensor_shape = None
        self.testing_phase = False
        self.trainable = True
        # CHANNELS - is the number of channels in the input tensor
        self.channels = channels
        self.weights = None
        self.bias = None
        self.initialize(channels)

        self._optimizer = None
        self.moving_mean = 0
        self.moving_variance = 0
        self.input_tensor = None
        self.normalized_input_tensor = None

    def initialize(self, shape):
        self.weights = np.ones(shape)
        self.bias = np.zeros(shape)

    def forward(self, input_tensor, alpha=0.8):
        self.tensor_shape = input_tensor.shape
        self.input_tensor = input_tensor
        # moving average provides better estimate of the true mean and variance compared to individual mini-batch
        # https://stackoverflow.com/questions/60460338/does-batchnormalization-use-moving-average-across-batches-or-only-per-batch-and
        y_bar = None
        eps = np.finfo(float).eps
        if len(input_tensor.shape) == 4:
            # if Convolution layer gives output with multiple dimensions ( batch, no. of kernel, height of image, width of image)
            self.mean = np.mean(input_tensor, axis=(0, 2, 3))
            self.variance = np.var(input_tensor, axis=(0, 2, 3))
            no_of_kernels = input_tensor.shape[1]
            if not self.testing_phase:
                # Training Phase
                mean = np.mean(input_tensor, axis=(0, 2, 3))
                variance = np.var(input_tensor, axis=(0, 2, 3))
                self.moving_mean = alpha * self.mean + (1-alpha) * mean
                self.moving_variance = alpha * self.variance + (1-alpha)*variance
                self.mean = mean
                self.variance = variance
                self.normalized_input_tensor = (input_tensor - self.mean.reshape((1, no_of_kernels, 1, 1))) / np.sqrt(self.variance.reshape((1, no_of_kernels, 1, 1)) + eps)
                y_bar = self.weights.reshape((1, no_of_kernels, 1, 1)) * self.normalized_input_tensor + self.bias.reshape((1, no_of_kernels, 1, 1))
            else:
                # Testing Phase
                self.normalized_input_tensor = (input_tensor - self.moving_mean.reshape((1, no_of_kernels, 1, 1))) / np.sqrt(self.moving_variance.reshape((1, no_of_kernels, 1, 1)) + eps)
                y_bar = self.weights.reshape((1, no_of_kernels, 1, 1)) * self.normalized_input_tensor + self.bias.reshape((1, no_of_kernels, 1, 1))

        else:
            # if only batch number and number of kernels comes out
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)
            if not self.testing_phase:
                # Training Phase
                mean = np.mean(input_tensor, axis=0)
                variance = np.var(input_tensor, axis=0)
                self.moving_mean = alpha * self.mean + (1 - alpha) * mean
                self.moving_variance = alpha * self.variance + (1 - alpha) * variance
                self.mean = mean
                self.variance = variance
                self.normalized_input_tensor = (input_tensor - self.mean) / np.sqrt(self.variance + eps)
                y_bar = self.weights * self.normalized_input_tensor + self.bias
            else:
                # Testing Phase
                self.normalized_input_tensor = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + eps)
                y_bar = self.weights * self.normalized_input_tensor + self.bias
        return y_bar

    def backward(self, error_tensor):
        eps = np.finfo(float).eps
        if len(error_tensor.shape) == 4:
            output = compute_bn_gradients(error_tensor = self.reformat(error_tensor),
                                          input_tensor=self.reformat(self.input_tensor),
                                          weights=self.weights,
                                          mean=self.mean,
                                          var=self.variance)
            output = self.reformat(output)
            self.gradient_weights = np.sum(error_tensor * self.normalized_input_tensor, axis=(0, 2, 3))
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        else:
            output = compute_bn_gradients(error_tensor = self.reformat(error_tensor),
                                          input_tensor=self.reformat(self.input_tensor),
                                          weights=self.weights,
                                          mean=self.mean,
                                          var=self.variance)
            self.gradient_weights = np.sum(error_tensor * self.normalized_input_tensor, axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)
        return output

    def reformat(self, tensor):
        if len(tensor.shape) > 2:
            # this will run when there are 4 dimensions
            tensor_shape = tensor.shape
            tensor = tensor.reshape(tensor_shape[0], tensor_shape[1], np.prod(tensor_shape[2:]))
            tensor = np.swapaxes(tensor, 1, tensor.ndim-1)
            # reshape to B.M.N x H
            tensor = tensor.reshape(-1, tensor_shape[1])
            return tensor
        else:
            # reverse of "if" is here
            tensor_shape = self.input_tensor.shape
            tensor = tensor.reshape(tensor_shape[0], -1, tensor_shape[1])
            tensor = np.swapaxes(tensor, 1, tensor.ndim-1)
            tensor = tensor.reshape(tensor_shape[0], tensor_shape[1], *(tensor_shape[2:]))
            return tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @gradient_bias.deleter
    def gradient_bias(self):
        del self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value

    @bias_optimizer.deleter
    def bias_optimizer(self):
        del self._bias_optimizer
