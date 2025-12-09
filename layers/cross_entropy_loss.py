import numpy as np


class CrossEntropyLoss:
    """
    Calculates Cross-Entropy Loss and the simplified gradient
    (dL/dZ) for Softmax inputs.
    """

    def forward(self, activation_output, targets):
        """
        Calculates the Cross-Entropy Loss L.
        - activation_output (A_hat): Probabilities from Softmax (B, C)
        - targets (Y): One-hot encoded true labels (B, C)
        """
        self.activation_output = activation_output
        self.targets = targets

        # Add a tiny epsilon to probabilities to prevent log(0), ensuring numerical stability
        epsilon = 1e-10
        A_hat = activation_output + epsilon

        # L = - sum(y * log(y_hat))
        # We take the mean loss over the entire batch
        loss = -np.sum(targets * np.log(A_hat)) / targets.shape[0]

        return loss

    def derivative(self):
        """
        Calculates the simplified gradient of the loss with respect to the
        raw input scores (Z) to the Softmax layer: dL/dZ = A_hat - Y.

        Returns a gradient tensor of shape (B, C).
        """
        # dL/dZ = A_hat - Y (The famous simplified gradient)
        # This gradient is averaged across the batch size (B) during backprop
        d_L_d_Z = self.activation_output - self.targets

        # The gradient must be scaled down by the batch size,
        # since the loss was averaged over the batch in the forward pass.
        d_L_d_Z_normalized = d_L_d_Z / self.targets.shape[0]

        return d_L_d_Z_normalized
