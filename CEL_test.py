import numpy as np
from layers.cross_entropy_loss import CrossEntropyLoss

# Dummy data: Batch size 2, 3 classes
BATCH_SIZE = 2
targets = np.array([[0, 1, 0], [1, 0, 0]])  # One-hot encoded (B, C)

# Dummy Softmax output (must sum to 1.0 on axis 1)
# Prediction 1: (0.1, 0.8, 0.1) -> Loss should be low (correctly predicts class 1)
# Prediction 2: (0.6, 0.2, 0.2) -> Loss should be higher (correctly predicts class 0, but with low confidence)
activation_output = np.array([[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])

loss_layer = CrossEntropyLoss()

# Forward Test
loss = loss_layer.forward(activation_output, targets)
# Expected loss is approx 0.356 (check calculation: [-log(0.8) + -log(0.6)] / 2)
print(f"Calculated Loss: {loss:.4f}")

# Backward Test
d_L_d_Z = loss_layer.backward()

print(f"Gradient dL/dZ shape: {d_L_d_Z.shape}")
print(f"d_L_d_Z (Normalized by B={BATCH_SIZE}):\n{d_L_d_Z}")

# dL/dZ should be (A - Y) / B
# Sample 1: ([0.1, 0.8, 0.1] - [0, 1, 0]) / 2 = [0.1, -0.2, 0.1] / 2 = [0.05, -0.1, 0.05]
# Sample 2: ([0.6, 0.2, 0.2] - [1, 0, 0]) / 2 = [-0.4, 0.2, 0.2] / 2 = [-0.2, 0.1, 0.1]
