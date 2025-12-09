import numpy as np
from layers.pool2D import Pool2D

# Assuming your final Pool2D class is ready and imported/defined above
# ... (Pool2D class code) ...


def test_pool2d_layer():
    print("--- 🔬 Pool2D Comprehensive Test ---")

    # Simple input data: (B, H, W, D) = (1, 4, 4, 1)
    # We use easily recognizable values for manual gradient checking
    input_data = np.array(
        [
            [
                [[1.0], [2.0], [3.0], [4.0]],
                [[5.0], [6.0], [7.0], [8.0]],
                [[9.0], [10.0], [11.0], [12.0]],
                [[13.0], [14.0], [15.0], [16.0]],
            ]
        ],
        dtype=np.float32,
    )

    B, H, W, D = input_data.shape
    LEARN_RATE = 0.01  # Not used, but included for backprop signature

    # Incoming gradient dL/dA: A simple 1.0 gradient for easy tracking
    grad_in_local = np.array([[[[1.0], [1.0]], [[1.0], [1.0]]]])

    # --- 1. LOCAL MAX POOLING (2x2, Stride 2) ---
    print("\n## 1. Local Max Pooling (2x2, Stride 2)")

    max_pool = Pool2D(pool_size=2, stride=2, mode="max")
    output = max_pool.forward(input_data)

    # Expected Output: Max of each 2x2 region (6, 8, 14, 16)
    expected_output_local = np.array([[[[6.0], [8.0]], [[14.0], [16.0]]]])

    # Check Forward Pass
    if np.allclose(output, expected_output_local):
        print("✅ Forward Pass Shape/Value: CORRECT")
    else:
        print(
            "❌ Forward Pass FAILED. Expected:\n",
            expected_output_local,
            "\nGot:\n",
            output,
        )

    # Check Backward Pass (Gradient should only flow to the 'max' locations: 6, 8, 14, 16)
    grad_out = max_pool.backprop(grad_in_local, LEARN_RATE)

    # Expected gradient mask (1.0 at max locations, 0.0 elsewhere)
    expected_grad_mask = np.array(
        [
            [
                [[0.0], [0.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0], [1.0]],
                [[0.0], [0.0], [0.0], [0.0]],
                [[0.0], [0.0], [1.0], [1.0]],
            ]
        ]
    )

    print(grad_out)
    if np.allclose(grad_out, expected_grad_mask):
        print("✅ Backward Pass Gradient Routing: CORRECT (Max locations only)")
    else:
        print("❌ Backward Pass FAILED. Check mask logic.")

    # --- 2. LOCAL AVERAGE POOLING (2x2, Stride 2) ---
    print("\n## 2. Local Average Pooling (2x2, Stride 2)")

    avg_pool = Pool2D(pool_size=2, stride=2, mode="average")
    output = avg_pool.forward(input_data)

    # Expected Output: Average of each 2x2 region (1+2+5+6)/4=3.5, (3+4+7+8)/4=5.5, etc.
    expected_output_local = np.array([[[[3.5], [5.5]], [[11.5], [13.5]]]])

    # Check Forward Pass
    if np.allclose(output, expected_output_local):
        print("✅ Forward Pass Shape/Value: CORRECT")
    else:
        print("❌ Forward Pass FAILED.")

    # Check Backward Pass (Gradient 1.0 is spread to all 4 locations: 1.0 / 4 = 0.25)
    grad_out = avg_pool.backprop(grad_in_local, LEARN_RATE)

    # Expected gradient: 0.25 in the 2x2 regions covered by the local pooling
    expected_grad_spread = np.array(
        [
            [
                [[0.25], [0.25], [0.25], [0.25]],
                [[0.25], [0.25], [0.25], [0.25]],
                [[0.25], [0.25], [0.25], [0.25]],
                [[0.25], [0.25], [0.25], [0.25]],
            ]
        ]
    )

    if np.allclose(grad_out, expected_grad_spread):
        print("✅ Backward Pass Gradient Spreading: CORRECT (0.25 to all)")
    else:
        print("❌ Backward Pass FAILED. Check spreading logic.")

    # --- 3. GLOBAL MAX POOLING (4x4 -> 1x1) ---
    print("\n## 3. Global Max Pooling (4x4 -> 1x1)")

    global_max_pool = Pool2D(
        mode="global_max"
    )  # pool_size and stride are irrelevant here
    output = global_max_pool.forward(input_data)

    # Expected Output: Max of the entire input (16)
    expected_output_global = np.array([[[[16.0]]]])

    if np.allclose(output, expected_output_global):
        print("✅ Forward Pass Shape/Value: CORRECT")
    else:
        print("❌ Forward Pass FAILED.")

    # Incoming gradient dL/dA: Simple 1.0 (Shape: 1, 1, 1, 1)
    grad_in_global = np.array([[[[1.0]]]])
    grad_out = global_max_pool.backprop(grad_in_global, LEARN_RATE)

    # Check Backward Pass (Gradient 1.0 should ONLY go to the location of 16.0)
    # Location of 16.0 is (0, 3, 3, 0)
    expected_grad_global_max = np.zeros_like(input_data)
    expected_grad_global_max[0, 3, 3, 0] = 1.0

    if np.allclose(grad_out, expected_grad_global_max):
        print("✅ Backward Pass Global Max Routing: CORRECT (Only at [3, 3])")
    else:
        print(
            "❌ Backward Pass FAILED. Check global max logic. Expected value 1.0 at [3, 3]."
        )

    # --- 4. GLOBAL AVERAGE POOLING (4x4 -> 1x1) ---
    print("\n## 4. Global Average Pooling (4x4 -> 1x1)")

    global_avg_pool = Pool2D(mode="global_average")
    output = global_avg_pool.forward(input_data)

    # Expected Output: Average of the entire input (1+2+...+16)/16 = 8.5
    expected_output_global = np.array([[[[8.5]]]])

    if np.allclose(output, expected_output_global):
        print("✅ Forward Pass Shape/Value: CORRECT")
    else:
        print("❌ Forward Pass FAILED.")

    # Check Backward Pass (Gradient 1.0 is spread to all 16 locations: 1.0 / 16 = 0.0625)
    grad_in_global = np.array([[[[1.0]]]])
    grad_out = global_avg_pool.backprop(grad_in_global, LEARN_RATE)

    expected_grad_global_avg = np.ones_like(input_data) * (1.0 / 16.0)

    if np.allclose(grad_out, expected_grad_global_avg):
        print("✅ Backward Pass Global Avg Spreading: CORRECT (0.0625 to all)")
    else:
        print("❌ Backward Pass FAILED. Check global average spreading.")


if __name__ == "__main__":
    test_pool2d_layer()
