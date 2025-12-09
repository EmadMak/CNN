import numpy as np
from Activations import ReLU, SiLU, softmax


class Dense:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation  # Store activation object

        # Weights (W): Shape (input_size, output_size)
        # Change from 2.0 to 1.0 to switch between He initialization and Xavier
        std_dev = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * std_dev

        # Biases (B): Shape (1, output_size) - Batch dimension is handled by broadcasting
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        """
        Performs: Z = X * W + B, followed by A = f(Z)
        - input is a 2D numpy array (Batch Size, input_size).
        """
        self.last_input = input

        # 1. Linear transformation (Z)
        output_raw = np.dot(input, self.weights) + self.biases
        self.last_output_raw = output_raw  # Store Z before activation

        # 2. Apply Activation (A)
        output_activated = self.activation.forward(output_raw)
        return output_activated

    def backprop(self, d_L_d_out, learn_rate):
        """
        Performs the backward pass.
        - d_L_d_out is the gradient of the Loss w.r.t the activated output (dL/dA).
        """

        # 1. Backpropagate through the activation function
        # d_L_d_conv_out is dL/dZ (Loss gradient w.r.t raw output)
        d_L_d_conv_out = self.activation.derivative(d_L_d_out)

        # 2. Gradient w.r.t. Weights (W): dL/dW = X^T * dL/dZ
        d_L_d_W = np.dot(self.last_input.T, d_L_d_conv_out)

        # 3. Gradient w.r.t. Biases (B): dL/dB = Sum(dL/dZ, axis=0)
        # Sum across the batch dimension (axis 0)
        d_L_d_B = np.sum(d_L_d_conv_out, axis=0, keepdims=True)

        # 4. Gradient w.r.t. Input (X): dL/dX = dL/dZ * W^T
        d_L_d_input = np.dot(d_L_d_conv_out, self.weights.T)

        # 5. Update parameters (SGD)
        self.weights -= learn_rate * d_L_d_W
        self.biases -= learn_rate * d_L_d_B

        # 6. Return the input gradient
        return d_L_d_input
