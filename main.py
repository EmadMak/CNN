import os

import numpy as np
from Activations import ReLU, SiLU

# Assuming your classes are defined in separate files or correctly defined here
from layers.conv2D_2 import Conv2D
from layers.cross_entropy_loss import CrossEntropyLoss
from layers.dense import Dense
from layers.flatten import Flatten
from layers.pool2D import Pool2D
from neural_network import NeuralNetwork
from PIL import Image  # Use PIL for image handling

# --- 1. Data Utility Functions ---


def create_dummy_data(num_samples=100, H=28, W=28, D=1, num_classes=10):
    """Creates a simple, random dataset for testing the network structure."""

    # Input Data: (B, H, W, D) -> (100, 28, 28, 1)
    X = np.random.rand(num_samples, H, W, D).astype(np.float32)

    # True Labels (Indices): (B,)
    y_indices = np.random.randint(0, num_classes, num_samples)

    # One-Hot Encoded Targets (Y): (B, C) -> (100, 10)
    Y_true = np.zeros((num_samples, num_classes))
    Y_true[np.arange(num_samples), y_indices] = 1

    return X, Y_true, y_indices


def accuracy(y_pred_indices, y_true_indices):
    """Calculates the classification accuracy."""
    return np.mean(y_pred_indices == y_true_indices)


# --- 2. Model Definition ---


def define_cnn_model():
    """Defines the sequential layers of the CNN architecture."""
    # Assuming a 28x28x1 input (like MNIST)
    input_depth = 1
    num_classes = 10

    FINAL_FEATURE_MAP_DEPTH = 16

    layers = [
        # --- 1. Conv Layer (Maintains resolution: 28x28x8) ---
        Conv2D(
            kernel_number=8,
            input_depth=input_depth,
            activation=ReLU(),
            padding_mode="same",
        ),
        # --- 2. Conv Layer (Maintains resolution: 28x28x16) ---
        Conv2D(
            kernel_number=FINAL_FEATURE_MAP_DEPTH,
            input_depth=8,
            activation=ReLU(),
            padding_mode="same",
        ),
        # --- 3. Global Average Pool (Reduces size: 28x28x16 -> 1x1x16) ---
        Pool2D(mode="global_average"),
        Flatten(),
        # --- 4. Dense Layer 1 ---
        # Input size MUST be the depth of the feature map (16)
        Dense(input_size=FINAL_FEATURE_MAP_DEPTH, output_size=64, activation=ReLU()),
        # --- 5. Dense Layer 2 (Output Layer) ---
        Dense(input_size=64, output_size=num_classes, activation=ReLU()),
    ]

    return NeuralNetwork(layers=layers)


def load_data(base_dir="./data/augmented/", target_size=(19, 19)):
    """
    Loads ALL PNG images recursively under base_dir, extracting the class label
    (0-9) from the filename's first character.
    """
    X_data = []
    Y_labels = []
    num_classes = 10  # Explicitly set to 10 classes (0-9)

    # 1. Traverse the entire directory tree under base_dir
    # root: current path
    # dirs: list of directories in the current path
    # files: list of files in the current path
    for root, dirs, files in os.walk(base_dir):
        # 2. Iterate through all files found in the current directory (root)
        for filename in files:
            if filename.endswith(".png"):
                img_path = os.path.join(root, filename)

                # --- CRITICAL CHANGE: Extract label from filename ---
                try:
                    # Filename format: X_Y_... .png (We want X)
                    class_label = int(filename[0])

                    # Ensure the label is within the 0-9 range
                    if not (0 <= class_label < num_classes):
                        continue

                except (ValueError, IndexError):
                    # Skip files that don't start with a number or are empty
                    continue

                # --- Image Loading and Preprocessing ---
                try:
                    img_pil = Image.open(img_path)
                    img_pil = img_pil.convert("L").resize(target_size)
                    img = np.array(img_pil, dtype=np.float32) / 255.0
                    img = np.expand_dims(img, axis=-1)

                    X_data.append(img)
                    Y_labels.append(class_label)

                except Exception as e:
                    # Only print errors for debugging, not during normal run
                    # print(f"Skipping file due to loading error: {img_path}. Error: {e}")
                    continue

    # Final Conversion and One-Hot Encoding
    # ... (Rest of the function remains the same) ...
    X_data = np.array(X_data)
    y_indices = np.array(Y_labels)

    Y_true = np.zeros((len(y_indices), num_classes))
    Y_true[np.arange(len(y_indices)), y_indices] = 1

    print(
        f"Loaded {len(X_data)} images for {num_classes} classes (0-9). Final shape: {X_data.shape}"
    )

    return X_data, Y_true, y_indices, num_classes


def split_data(X, Y, y_indices, test_ratio=0.2):
    """Splits data into training and testing sets using shuffled indices."""

    size = X.shape[0]

    # 1. Create a full permutation of indices
    full_indices = np.random.permutation(size)

    # 2. Determine the split point
    test_size = int(size * test_ratio)  # e.g., 6250

    # 3. Define the slices based on the split point
    test_indices = full_indices[:test_size]  # First 6250 indices
    train_indices = full_indices[test_size:]  # Remaining indices

    # --- Apply Indices to Data ---

    # Testing Set
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    y_test_indices = y_indices[test_indices]

    # Training Set
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    y_train_indices = y_indices[train_indices]

    # Verification Print (Add this to your main.py to confirm the split)
    print(f"Data Split: Total={size}, Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    return X_train, Y_train, y_train_indices, X_test, Y_test, y_test_indices


# --- 3. Main Execution Function ---


def main(mode="train"):
    BASE_DIR = os.path.join("..", os.getcwd(), "data", "augmented")

    X_all, Y_all, y_all_indices, num_classes = load_data(BASE_DIR)

    # Split the data into train and test sets
    X_train, Y_train, y_train_indices, X_test, Y_test, y_test_indices = split_data(
        X_all, Y_all, y_all_indices, test_ratio=0.2
    )

    # Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 64
    LEARN_RATE = 0.01
    PARAM_FILE = "trained_cnn_params.npz"

    model = define_cnn_model()

    if mode == "train":
        print("--- Starting Training ---")

        # Train the model
        model.fit(
            X_train=X_train,
            Y_train=Y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learn_rate=LEARN_RATE,
        )

        # Save the trained parameters
        model.save_params(PARAM_FILE)

        # Evaluate on test set
        y_pred_indices = model.predict(X_test)
        acc = accuracy(y_pred_indices, y_test_indices)
        print(f"\nTraining Complete. Final Test Accuracy: {acc * 100:.2f}%")

    elif mode == "predict":
        print("--- Starting Inference from Saved Parameters ---")

        # Load parameters into the model
        model.load_params(PARAM_FILE)

        # Predict on test set
        y_pred_indices = model.predict(X_test)
        acc = accuracy(y_pred_indices, y_test_indices)
        print(f"Inference Complete. Test Accuracy (Loaded Model): {acc * 100:.2f}%")

    else:
        print("Invalid mode. Use 'train' or 'predict'.")


if __name__ == "__main__":
    # Run the full training and evaluation sequence
    main(mode="train")

    # If you want to test loading, uncomment the line below after running train once:
    # main(mode="predict")
