import numpy as np
from Activations import ReLU, SiLU, Softmax
from layers.cross_entropy_loss import CrossEntropyLoss
from layers.dense import Dense


def test_dense_network():
    print("--- Dense Layer Test Execution ---")

    # 1. Setup Parameters
    BATCH_SIZE = 4
    INPUT_SIZE = 50
    HIDDEN_SIZE = 20
    OUTPUT_CLASSES = 10
    LEARN_RATE = 0.01

    # Dummy Input: (B, I)
    input_data = np.random.randn(BATCH_SIZE, INPUT_SIZE)
    # Dummy Targets: (B, C) - One-hot encoded labels
    targets = np.zeros((BATCH_SIZE, OUTPUT_CLASSES))
    targets[np.arange(BATCH_SIZE), np.random.randint(0, OUTPUT_CLASSES, BATCH_SIZE)] = (
        1.0
    )

    print(f"Input data shape: {input_data.shape}")
    print(f"Target labels shape: {targets.shape}\n")

    # 2. Initialize Layers
    dense_hidden = Dense(INPUT_SIZE, HIDDEN_SIZE, activation=ReLU())
    dense_output = Dense(HIDDEN_SIZE, OUTPUT_CLASSES, activation=Softmax())
    loss_layer = CrossEntropyLoss()

    initial_W1 = dense_hidden.weights.copy()
    initial_B1 = dense_hidden.biases.copy()
    initial_W2 = dense_output.weights.copy()
    initial_B2 = dense_output.biases.copy()

    # --- FORWARD PASS ---
    print("--- 3. Forward Pass (Test Shapes) ---")

    hidden_output = dense_hidden.forward(input_data)
    final_output = dense_output.forward(hidden_output)

    print(f"Hidden output shape: {hidden_output.shape}")
    print(f"Final output shape:  {final_output.shape}")

    # Check Softmax: all probabilities should be positive and sum to 1
    if np.allclose(np.sum(final_output, axis=1), 1.0) and np.all(final_output >= 0):
        print("✅ Forward Pass: Shapes and Softmax OK.")
    else:
        print("❌ Forward Pass: FAILED Softmax Check.")

    loss = loss_layer.forward(final_output, targets)
    print(f"Calculated Loss: {loss:.4f}\n")

    # --- BACKWARD PASS ---
    print("--- 4. Backward Pass (Test Gradients and Updates) ---")

    # 1. Calculate the final gradient dL/dZ from the Loss Layer
    d_L_d_Z_output = loss_layer.backward()

    # 2. Backprop through Layer 2 (Output)
    # d_L_d_Z_output is passed as d_L_d_out. Softmax.backward returns it unchanged.
    d_L_d_hidden_output = dense_output.backprop(d_L_d_Z_output, LEARN_RATE)

    # 3. Backprop through Layer 1 (Hidden)
    # The gradient d_L_d_hidden_output is passed back. ReLU.backward takes the derivative.
    d_L_d_input = dense_hidden.backprop(d_L_d_hidden_output, LEARN_RATE)

    # --- Verification Checks ---

    # Check 1: Final Input Gradient Shape
    if d_L_d_input.shape == input_data.shape:
        print(f"✅ Input Gradient Shape: {d_L_d_input.shape} (Matches input)")
    else:
        print("❌ Input Gradient Shape: FAILED")

    # Check 2: Weight Updates
    def check_update(initial, current, name):
        if not np.array_equal(initial, current):
            print(f"✅ {name}: Parameters UPDATED.")
        else:
            print(f"❌ {name}: Parameters NOT UPDATED.")

    check_update(initial_W2, dense_output.weights, "Layer 2 Weights")
    check_update(initial_B2, dense_output.biases, "Layer 2 Biases")
    check_update(initial_W1, dense_hidden.weights, "Layer 1 Weights")
    check_update(initial_B1, dense_hidden.biases, "Layer 1 Biases")


if __name__ == "__main__":
    test_dense_network()
