import numpy as np
from Layers.Base import BaseLayer
import math


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        # Stride is the number of pixels that kernel can skip
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.indexing_list = None
        self.input_shape = None
        self.parameters = None
        self.output_tensor = None

        pass

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        batch_size, input_depth, image_height, image_width = self.input_shape
        # Calculating the output size here bbbbbb
        kernel_height = math.floor(1 + (image_height - self.pooling_shape[0]) / self.stride_shape[0])
        kernel_width = math.floor(1 + (image_width - self.pooling_shape[1]) / self.stride_shape[1])
        self.output_tensor = np.zeros((batch_size, input_depth, kernel_height, kernel_width))
        self.parameters = []
        self.indexing_list = []

        for batch in range(batch_size):
            for depth in range(input_depth):
                for height in range(kernel_height):
                    for width in range(kernel_width):
                        var = input_tensor[batch, depth,
                                           height * self.stride_shape[0]:height * self.stride_shape[0] + self.pooling_shape[0],
                                           width * self.stride_shape[1]: width * self.stride_shape[1] + self.pooling_shape[1]]
                        if var.size > 0:
                            max_val = np.max(var)
                            self.output_tensor[batch, depth, height, width] = max_val
                            max_index = tuple(np.squeeze(np.where(var == max_val)))
                            self.indexing_list.append((batch, depth, max_index[0] + height*self.stride_shape[0], max_index[1] +  width * self.stride_shape[1] ))
                            self.parameters.append((batch, depth, height, width))
        return self.output_tensor

    def backward(self, error_tensor):
        output = np.zeros(self.input_shape)
        for index, location in enumerate(self.indexing_list):
            output[location] += error_tensor[self.parameters[index]]
        return output
