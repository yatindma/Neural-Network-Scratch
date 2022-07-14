from scipy.signal import correlate, convolve
import numpy as np
from Layers.Base import BaseLayer
from copy import deepcopy



class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        # Determine the dimensionality of the convolution for later simplification
        self.one_dimension_flag = False
        self.num_kernels = num_kernels



        # 1D convolution
        if len(convolution_shape) == 2:
            # adding arbitrary dimension to make it compatible later
            self.convolution_shape += (1,)
            self.one_dimension_flag = True
            self.stride_shape += (1,)


        # 2D convolution
        elif len(convolution_shape) == 3:
            # Repeat the stride shape to make it consistent
            if len(stride_shape) == 1:
                self.stride_shape = (stride_shape, stride_shape)

        # Initialize Weights and bias
        self.weights = np.random.uniform(size=(self.num_kernels,)+self.convolution_shape)  # (num_kernels, c, m, n)
        # we have to have bias for each kernel - hence initializing it here with random normal values
        self.bias = np.random.rand(self.num_kernels)

        self.gradient_weights = None
        self.gradient_bias = None
        self.input_tensor_padded = None
        self.output_shape_intact = None
        self.padding = None
        self.padding_list = None
        self._optimizers = None
        self.input_tensor = None

        self.input_size = None
        self.input_pad_size = None

    def forward(self, input_tensor):
        self.input_size = input_tensor.shape
        self.input_tensor = input_tensor

        # Add one more dimension to make it compatible
        if self.one_dimension_flag:
            self.input_size += (1,) # adding size to it to make it compatible later with 2D code
            self.input_tensor = input_tensor[:, :, :, np.newaxis]


        # Output shape if there is no stride (batch size, Number of Kernels, height and width)
        self.output_shape_intact = (self.input_size[0], self.num_kernels,
                                    self.input_size[2], self.input_size[3])

        # if filter is of NxN dimension then we need to add zero-padding and it'll be asymetric
        self.padding = (self.convolution_shape[1] - 1, self.convolution_shape[2] - 1)
        self.padding_list = [(0, 0), (0, 0),
                        (np.ceil(self.padding[0]/2).astype(int), np.floor(self.padding[0]/2).astype(int)),
                        (np.ceil(self.padding[1]/2).astype(int), np.floor(self.padding[1]/2).astype(int))]
        # adding padding to the input tensor
        self.input_pad_size = self.input_tensor_padded.shape
        self.input_tensor_padded = np.pad(self.input_tensor, self.padding_list, mode="constant", constant_values=0)



        result_nonstrided = np.zeros(self.output_shape_intact)
        # we are calculating the correlation in forward pass and will do the convolution in the backward
        # adding constant bias in the result
        for i in range(self.output_shape_intact[0]):
            for j in range(self.num_kernels):
                result_nonstrided[i, j, :, :] = correlate(self.input_tensor_padded[i, :, :, :],
                                                      self.weights[j, :, :, :], mode='valid') + self.bias[j]

        # adding the strides to the non-strided matrix  .....
        result = result_nonstrided[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        # Special 1D case
        if self.one_dimension_flag : result = result[:, :, :, 0]

        return result

    def backward(self, error_tensor):
        if self.one_dimension_flag:
            error_tensor = error_tensor[:, :, :, np.newaxis]

        # Initialize final variables
        gradient_input   = np.zeros(self.input_size)
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)



        # Upsample and pad error tensor
        err_ups = np.zeros(self.output_shape_intact)
        err_ups[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor # De-striding literally
        err_pad = np.pad(err_ups, self.padding_list, mode="constant", constant_values=0)  # same pad as forward

        # 1. Compute gradient wrt lower layers --> Compute the error of this layer
        # Combine kernels (c kernels of H depth - each layer from each original kernel - in order)
        new_ker_shape = (self.convolution_shape[0], self.num_kernels,
                         self.convolution_shape[1], self.convolution_shape[2])

        new_ker_w = np.zeros(new_ker_shape)

        for c in range(self.convolution_shape[0]):
            for k in range(self.num_kernels):
                # Combine kernels --> c_in filters with c_out channels each
                new_ker_w[c, k, :, :] = self.weights[k, c, :, :]

            # Perform convolution --> Forward was cross-correlation
            for b in range(self.input_size[0]): # Batch size
                # Flip up-down,  already flips the kernels left-right ...... ... .
                flipped = np.flipud(new_ker_w[c, :, :, :])
                gradient_input[b, c, :, :] = convolve(err_pad[b, :, :, :], flipped, mode="valid")

        # 2. Compute gradient wrt to bias: sum over each channel of error
        self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))  # (K,) - We do not sum over channel axis  .......

        # 3. Compute gradient wrt weights
        for b in range(self.input_size[0]):  # loop for every element of the batch
            for c in range(self.input_size[1]):  # loop for every input channel ..... ....
                for k in range(self.num_kernels):  # loop for every output channel
                    # Correlation with upsampled (not padded) error
                    self.gradient_weights[k, c, :, :] += correlate(self.input_tensor_padded[b, c, :, :],
                                                                   err_ups[b, k, :, :], mode="valid")

        # Update step
        if self._optimizers is not None:
            self.bias = self._optimizers[1].calculate_update(self.bias, self.gradient_bias)
            self.weights = self._optimizers[0].calculate_update(self.weights, self.gradient_weights)


        if self.one_dimension_flag: gradient_input = gradient_input[:, :, :, 0]

        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        fan_in  = np.prod(list(self.convolution_shape))  # c * m * n
        fan_out = np.prod([self.num_kernels]+list(self.convolution_shape[1:]))  # k * m * n
        self.bias = bias_initializer.initialize(self.bias.shape, 1, fan_out)
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)

    @property
    def optimizer(self):
        return self._optimizers

    @optimizer.getter
    def optimizer(self):
        return self._optimizers

    @optimizer.setter
    def optimizer(self, optimizer): # Need to a second copy of the optimizer instance
        self._optimizers = [optimizer, deepcopy(optimizer)]