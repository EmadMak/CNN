import numpy as np

# from Activations.functions import Activations
# from ..Activations import ReLU, SiLU
from Activations import ReLU, SiLU
from scikit_clone.measure import *


class Conv2D:
    def __init__(self, kernel_number, kernel_size, activation, input_channels=1):
        self.kernel_number = kernel_number
        self.kernel_size = kernel_size
        self.kernels = self.initialize_kernels(
            kernel_number, kernel_size, input_channels
        )
        self.activation = activation
        self.input_channels = input_channels

    def initialize_kernels(
        self, kernel_number: int, kernel_size: int, input_channels: int = 1
    ):
        # Calculate Fan-in (n_in): kernel_height * kernel_width * input_channels
        fan_in = kernel_size * kernel_size * input_channels

        # Calculate the standard deviation for the scaling factor
        std_dev = np.sqrt(2.0 / fan_in)

        # Initialize kernels from a standard normal distribution and scale them
        # Shape: (C_out, K, K)
        kernels = (
            np.random.randn(kernel_number, input_channels, kernel_size, kernel_size)
            * std_dev
        )

        return kernels

    def forward(self, inputs: np.ndarray):
        """
        inputs: D x H x W  (grayscale)
        self.kernels: num_kernels x kH x kW
        """
        self.inputs = inputs
        H_out = inputs.shape[1] - self.kernels.shape[2] + 1
        W_out = inputs.shape[2] - self.kernels.shape[3] + 1
        self.outputs = np.zeros((self.kernels.shape[0], H_out, W_out))

        for idx_k, kernel in enumerate(self.kernels):
            # print(kernel.shape)
            self.outputs[idx_k] = correlate2d(inputs, kernel, padding=0)

        self.outputs = self.activation.forward(self.outputs)
        return self.outputs

    def update_gradients(self, nodeValues):
        print("kernels shape before updating: ", self.kernels.shape)
        nodeValuesWithActDerivative = nodeValues * self.activation.derivative(
            self.outputs
        )

        self.kernels_gradients = np.zeros_like(self.kernels)
        inputs_gradients = np.zeros_like(self.inputs)

        for idx_k, kernel in enumerate(self.kernels):
            # gradient w.r.t kernel
            self.kernels_gradients[idx_k] = correlate2d(
                self.inputs, nodeValuesWithActDerivative[idx_k], padding=0
            )

            # gradient w.r.t input
            inputs_gradients += convolve2d_multi_filter(
                nodeValuesWithActDerivative[idx_k], kernel, padding=kernel.shape[0] - 1
            )

        return inputs_gradients

    #    def update_gradients(self, nodeValues):
    #        # Apply activation derivative
    #        nodeValuesWithActDerivative = nodeValues * self.activation.derivative(
    #            self.outputs
    #        )
    #
    #        # Initialize gradients
    #        self.kernels_gradients = np.zeros_like(
    #            self.kernels
    #        )  # shape: (K, K_h, K_w, C_in)
    #        inputs_gradients = np.zeros_like(self.inputs)  # shape: (H, W, C_in)
    #
    #        print("shape on input: ", self.inputs.shape)
    #        # For each filter
    #        for k_idx, kernel in enumerate(self.kernels):
    #            # Gradient w.r.t kernel: correlate input with delta per channel
    #            for c in range(self.inputs.shape[2]):  # iterate over input channels
    #                self.kernels_gradients[k_idx, :, :, c] = correlate2d(
    #                    self.inputs[c, :, :], nodeValuesWithActDerivative[k_idx], padding=0
    #                )
    #
    #            # Gradient w.r.t input: sum over convolutions with flipped kernel
    #            for c in range(self.inputs.shape[2]):
    #                inputs_gradients[:, :, c] += convolve2d_multi_filter(
    #                    nodeValuesWithActDerivative[k_idx],
    #                    np.flip(kernel[:, :, c], axis=(0, 1)),
    #                )
    #
    #        return inputs_gradients

    def apply_gradients(self, learn_rate):
        self.kernels -= learn_rate * self.kernels_gradients

    def reset_gradients(self):
        self.kernels_gradients = np.zeros_like(self.kernels)
