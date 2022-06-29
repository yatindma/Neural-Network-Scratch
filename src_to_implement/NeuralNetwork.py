from Layers import *
from Optimization import *
import copy


class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self._testing_phase = None
        self.optimizer = optimizer
        self.input_tensor = None
        self.data_layer = None
        self.loss_layer = None
        self.layer = None
        self.loss = []
        self.layers = []
        self.input_tensor, self.label_tensor = None, None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._train = None
        self._test = None
        self._phase = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        self.layer = self.layers[0].forward(self.input_tensor)
        for itr_layer in self.layers[1:]:
            self.layer = itr_layer.forward(self.layer)
        self.layer = self.loss_layer.forward(self.layer, self.label_tensor)
        return self.layer

    def backward(self):
        self.layer = self.loss_layer.backward(self.label_tensor)
        for itr_layer in reversed(self.layers):
            self.layer = itr_layer.backward(self.layer)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for val in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        self.layer = self.layers[0].forward(input_tensor)
        for itr_layer in self.layers[1:]:
            self.layer = itr_layer.forward(self.layer)
        return self.layer

    @property
    def testing_phase(self):
        return self._testing_phase

    @testing_phase.getter
    def testing_phase(self):
        return self._testing_phase

    @testing_phase.setter
    def testing_phase(self, testing_phase):  # Need to a second copy of the optimizer instance
        self._testing_phase = testing_phase
