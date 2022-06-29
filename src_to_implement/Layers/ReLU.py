from Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):

    def __init__(self) -> None:
        super().__init__()
        self.input_tensor = None
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.where(input_tensor < 0, 0, input_tensor)
    
    def backward(self, error_tensor):
        return np.where(self.input_tensor < 0, 0, error_tensor)