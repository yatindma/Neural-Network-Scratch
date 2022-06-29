from.Base import BaseLayer
import numpy as np


class Flatten(BaseLayer):
    def __int__(self):
        super().__init__()
        self.shape = (0, 0)

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return np.reshape(input_tensor, (input_tensor.shape[0], -1))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)
