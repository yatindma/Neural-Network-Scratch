import numpy as np


class BaseLayer:
    def __init__(self) -> None:
        self.trainable = False
        self.weights = np.array(1)
        self.testing_phase = False

    def forward(self, input_tensor):
        return

    def backward(self, error_tensor):
        return
