import numpy as np


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
        pass

    def calculate_gradient(self, weights):
        """
            Calculate sub-gradients on the weights needed for the optimization
        """
        return weights * self.alpha

    def norm(self, weights):
        """
            Calculate the norm enhanced loss
        """
        return self.alpha * np.sqrt((np.square(weights).sum()))


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
        pass

    def calculate_gradient(self, weights):
        """
            Calculate sub-gradients on the weights needed for the optimization
        """
        # because derivative of weights is 0 hence we multiplied it with lambda directly
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        """
            Calculate the norm enhanced loss
        """
        # Only considering the absolute value
        return self.alpha * np.abs(weights).sum()
