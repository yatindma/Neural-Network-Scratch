

class L2_Regularizer:
    def __init__(self, alpha):
        self.weights = None
        self.alpha = alpha
        pass

    def calculate_gradient(self, weights):
        """
            Calculate sub-gradients on the weights needed for the optimization
        """
        self.weights = weights

        return

    def norm(self, weights):
        """
            Calculate the norm enhanced loss
        """
        return


class L1_Regularizer:
    def __init__(self, alpha):
        self.weights = None
        self.alpha = alpha
        pass

    def calculate_gradient(self, weights):
        """
            Calculate sub-gradients on the weights needed for the optimization
        """
        self.weights = weights

        return

    def norm(self, weights):
        """
            Calculate the norm enhanced loss
        """
        return
