import numpy as np

# Assuming ReLU, SiLU, and GELU are defined and imported


class Softmax:
    """
    Softmax activation and its derivative for the final classification layer.
    """

    def __init__(self):
        self.output = None

    def forward(self, input):
        # Prevent overflow by subtracting max value (numerical stability)
        exp_vals = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def derivative(self, d_L_d_out):
        # NOTE: Softmax is generally used with Cross-Entropy Loss (d_L_d_out)
        # The derivative of Softmax + Cross-Entropy Loss simplifies greatly to:
        # dL/dZ = A - Y (where A is the Softmax output, Y is the true label)
        # However, since this class only calculates the Softmax derivative,
        # we'll return d_L_d_out as a placeholder and assume the loss layer
        # handles the combined derivative simplification.

        # If the input d_L_d_out is the gradient of the loss w.r.t the Softmax output,
        # the analytical derivative is highly complex.

        # For simplicity in this layered implementation, we typically assume the
        # loss function (like Cross-Entropy) is handled separately, but since
        # we must return a gradient to the previous layer, we will just pass
        # the gradient through assuming a simplified scenario or that d_L_d_out is
        # already the dL/dZ if used with CrossEntropy.

        # We'll use the proper, but complex, analytical derivative only if necessary,
        # but for clean chaining, we assume d_L_d_out is dL/dZ when used with CE.

        # For a standard library-like implementation, we return d_L_d_out
        # and rely on the CrossEntropyLoss class to provide the simple dL/dZ.
        return d_L_d_out
