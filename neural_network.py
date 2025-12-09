import numpy as np
from Activations import ReLU, SiLU, Softmax
from layers.conv2D_2 import Conv2D
from layers.cross_entropy_loss import CrossEntropyLoss
from layers.flatten import Flatten
from layers.pool2D import Pool2D


class NeuralNetwork:
    def __init__(self, layers):
        """
        Initializes the model with a list of layers.

        Args:
            layers (list): An ordered list of layer objects
                           (Conv2D, Pool2D, ReLU, Dense, etc.)
        """
        self.layers = layers
        # The loss function is not part of the layers list,
        # but is managed separately during training.
        self.loss_fn = CrossEntropyLoss()

    def save_params(self, filename):
        """
        Saves all learnable parameters (weights and biases) to a file.
        Uses np.savez_compressed for efficient storage.
        """
        params = {}
        for i, layer in enumerate(self.layers):
            # Check for Dense or Conv2D layers
            if hasattr(layer, "weights"):
                params[f"W{i}"] = layer.weights
            if hasattr(layer, "biases"):
                params[f"B{i}"] = layer.biases
            if hasattr(layer, "filters"):  # Conv2D uses 'filters' instead of 'weights'
                params[f"F{i}"] = layer.filters

        np.savez_compressed(filename, **params)
        print(f"Parameters saved successfully to {filename}")

    def load_params(self, filename):
        """
        Loads parameters from a file and sets them in the corresponding layers.
        """
        try:
            params = np.load(filename)
        except FileNotFoundError:
            print(f"❌ Error: Parameter file not found at {filename}")
            return

        for i, layer in enumerate(self.layers):
            # Load weights/filters
            if f"W{i}" in params:
                layer.weights = params[f"W{i}"]
            elif f"F{i}" in params:
                layer.filters = params[f"F{i}"]

            # Load biases
            if f"B{i}" in params:
                layer.biases = params[f"B{i}"]

        print(f"Parameters loaded successfully from {filename}")

    def forward(self, input):
        """
        Performs a forward pass through all layers.
        """
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train_step(self, X_batch, Y_batch, learn_rate):
        """
        Performs one full training step: forward pass, loss calculation,
        backward pass, and parameter updates.

        Args:
            X_batch (np.array): Batch of input data.
            Y_batch (np.array): Batch of one-hot encoded true labels.
            learn_rate (float): Learning rate for parameter updates.

        Returns:
            float: The calculated loss for the batch.
        """

        # --- 1. Forward Pass ---

        # The output of the last layer (Dense) is the raw scores (Z_final)
        # We need the Softmax output (A_final) for the loss calculation.
        Z_final = self.forward(X_batch)

        # Apply Softmax to get probabilities (A_final)
        # Assuming Softmax is not in self.layers but is used here
        A_final = Softmax().forward(Z_final)

        # Calculate Loss
        loss = self.loss_fn.forward(A_final, Y_batch)

        # --- 2. Backward Pass ---

        # a) Start gradient calculation with the simplified loss gradient (dL/dZ_final)
        # dL/dZ = A_hat - Y
        grad_out = self.loss_fn.derivative()

        # b) Iterate backward through all layers
        # Note: We skip the Softmax layer because its gradient is included in dL/dZ_final
        for layer in reversed(self.layers):
            # Pass the gradient backwards and let the layer update its parameters (if any)
            # 'derivative' is used for the activation function backprop call (dL/dZ)
            # 'backprop' is used for the layer backprop call (dL/dIn)

            # Use 'backprop' for Conv2D, Pool2D, Dense, Flatten, etc.
            # Use 'derivative' for ReLU/SiLU (which act as filters on the gradient)

            # Since all your layers/activations have a single backprop entry point:
            grad_out = layer.backprop(grad_out, learn_rate)

        return loss

    def fit(self, X_train, Y_train, epochs, batch_size, learn_rate):
        """
        Training loop for the entire dataset.
        """
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            Y_shuffled = Y_train[permutation]

            epoch_loss = 0

            # Iterate through batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]

                # Perform the training step
                batch_loss = self.train_step(X_batch, Y_batch, learn_rate)
                epoch_loss += batch_loss

                # Log the batch loss every 100 steps to monitor in-epoch progress
                step_num = i // batch_size + 1
                if step_num % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - Step {step_num} - Batch Loss: {batch_loss:.4f}"
                    )

            avg_loss = epoch_loss / (num_samples / batch_size)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    def predict(self, X_test):
        """
        Generates predictions for test data.
        """
        # 1. Get raw scores (Z_final)
        Z_final = self.forward(X_test)

        # 2. Apply Softmax to get probabilities
        A_final = Softmax().forward(Z_final)

        # 3. Return the index of the highest probability (the predicted class)
        return np.argmax(A_final, axis=1)
