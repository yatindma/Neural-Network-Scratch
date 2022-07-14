import numpy as np
import math


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_gradients(self, weights):
        return weights + self.regularizer


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate: float):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.vk = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.vk = self.momentum_rate * self.vk - self.learning_rate * gradient_tensor
        if self.regularizer:
            regularized_gradient = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * regularized_gradient
        weight_tensor = weight_tensor + self.vk
        return weight_tensor


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.vk = 0
        self.rk = 0
        self.pow = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.vk = self.mu * self.vk + (1-self.mu)*gradient_tensor
        self.rk = self.rho * self.rk + (1-self.rho) * gradient_tensor * gradient_tensor
        eps = np.finfo(float).eps
        vk_vector = self.vk/(1-np.power(self.mu, self.pow))
        rk_vector = self.rk / (1 - np.power(self.rho, self.pow))
        self.pow += 1
        if self.regularizer:
            regularized_gradient = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * regularized_gradient
        weight_tensor -= self.learning_rate * (vk_vector/(np.sqrt(rk_vector) + eps))
        return weight_tensor


class Sgd(Optimizer):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        pass

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            regularized_gradient = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor -= regularized_gradient * self.learning_rate
        return weight_tensor - self.learning_rate * gradient_tensor

