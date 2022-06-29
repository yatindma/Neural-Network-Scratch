import numpy as np
from .Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, keep_prob):
        super().__init__()
        self.trainable = False
        self.testing_phase = False
        self.keep_prob = keep_prob
        self.mask = None
        self.input_tensor = None
        self.d3 = None
        pass

    def forward(self, input_tensor):
        # When it's in training phase then only return the dropout result
        # Here we are initializing randomly some values from the input_tensor to zero
        if not self.testing_phase:
            self.input_tensor = input_tensor
            self.d3 = np.random.rand(*input_tensor.shape) < self.keep_prob
            a3 = input_tensor * self.d3
            # We are dividing the value of a3 with the keep probability because of the loss function layer
            a3 /= self.keep_prob
            return a3
        return input_tensor

    def backward(self, error_tensor):
        if not self.testing_phase:
            return (error_tensor * self.d3) / self.keep_prob
        return error_tensor
