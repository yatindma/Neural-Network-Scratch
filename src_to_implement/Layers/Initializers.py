import math
import numpy as np


class Constant:
    def __init__(self, const=0.1):
        self.const = const

    def initialize(self, weights_shape, fan_in, fan_out):
        values = np.ones(weights_shape) * self.const
        return values




class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = math.sqrt(2)/math.sqrt((fan_out + fan_in))
        return np.random.normal(size=weights_shape, scale=sigma)

class UniformRandom:
    def __int__(self):
        self.weights = None
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.random.uniform(size=weights_shape)
        return self.weights

class He:
    def __int__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out = None):
        sigma = np.sqrt(2) / np.sqrt(fan_in)
        return np.random.normal(size=weights_shape, scale=sigma)
