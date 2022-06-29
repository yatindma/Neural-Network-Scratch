from Layers.Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.input_tensor = None
        self.error_tensor = None
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # create an array of above size
        self.weights = np.random.uniform(1, size=(self.input_size+1, self.output_size))
        self._optimizer = None
        self.bias = None

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)
        self.weights[:-1, :] = weights
        self.weights[-1:, :] = bias

    def forward(self, input_tensor):
        '''
            Returns a tensor that serves as the
            input tensor for the next layer
            # This layer is just multiplying the weights with the input_tensor
        '''
        # input_tensor_biased = np.ones((input_tensor.shape[0], input_tensor.shape[1]+1))
        # input_tensor_biased[:, :-1] = input_tensor

        batch_size = input_tensor.shape[0]
        bias = np.ones((batch_size, 1))
        input_tensor_biased = np.concatenate((input_tensor, bias), axis=1)
        self.input_tensor = input_tensor_biased
        output_tensor = np.dot(input_tensor_biased, self.weights)
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return np.dot(self.input_tensor.T, self.error_tensor)

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        error_tensor_n__1 = np.dot(error_tensor, self.weights.T)
        self.error_tensor = error_tensor
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return error_tensor_n__1[:, :-1]






