import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        eps = np.finfo(float).eps
        loss = np.sum(np.where(label_tensor == 1, -1 * np.log(prediction_tensor + eps), 0))
        return loss

    def backward(self, label_tensor):
        eps = np.finfo(float).eps
        e_n = - (label_tensor/self.prediction_tensor + eps)
        return e_n
