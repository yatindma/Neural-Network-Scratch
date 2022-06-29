import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):

    def __init__(self) -> None:
        super().__init__()
        self.prediction_tensor = None
        self.input = None
        self.y_hat = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor = input_tensor - np.max(input_tensor)
        self.prediction_tensor = np.exp(input_tensor) / np.sum(np.exp(input_tensor), axis=1).reshape((input_tensor.shape[0], -1))
        return self.prediction_tensor

    def backward(self, error_tensor):
        mul = np.sum(error_tensor*self.prediction_tensor, axis=1)
        mul = mul.reshape(error_tensor.shape[0], 1)
        error_tensor_n__1 = self.prediction_tensor * (error_tensor - mul)
        return error_tensor_n__1


