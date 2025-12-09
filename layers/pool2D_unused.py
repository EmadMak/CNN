from multiprocessing import reducer
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np

# Define a type alias for the reduction function (e.g., np.max, np.mean)
ArrayReducer = Callable[[np.ndarray], float]

# --- Shared Utility Function (The Unified Forward Pass) ---


def _calculate_output_dims(
    X_shape: Tuple[int, int, int], size: int, stride: int
) -> Tuple[int, int, int]:
    """Calculates the output dimensions (D_out, H_out, W_out) for a given size and stride."""
    D_in, H_in, W_in = X_shape
    # Standard formula for output size (no padding assumption)
    H_out = (H_in - size) // stride + 1
    W_out = (W_in - size) // stride + 1

    # We maintain the depth (D) dimension
    return D_in, H_out, W_out


def _block_reduce_forward(
    X: np.ndarray,
    size: int,
    stride: int,
    reducer: ArrayReducer,
    state_saver: Callable[
        [int, int, int, np.ndarray, Tuple[int, int, int, int]], None
    ],  # Function to save indices/mask for backprop
) -> np.ndarray:
    """
    Unified forward pass logic for both Max and Average pooling.
    Handles window sliding based on size and stride, and applies the specified reducer.

    X: Input feature map (D, H, W).
    size: Pooling window size (K).
    stride: Stride (S).
    reducer: The function to apply to the window (np.max or np.mean).
    state_saver: A callback function used to store backpropagation information (only used by MaxPool).
    """
    D_in, H_in, W_in = X.shape
    D_out, H_out, W_out = _calculate_output_dims(X.shape, size, stride)

    Z = np.zeros((D_out, H_out, W_out), dtype=X.dtype)

    K, S = size, stride

    for d in range(D_in):
        for h_out in range(H_out):
            for w_out in range(W_out):
                # Calculate window boundaries based on stride (S)
                h_start, h_end = h_out * S, h_out * S + K
                w_start, w_end = w_out * S, w_out * S + K

                window = X[d, h_start:h_end, w_start:w_end]

                # 1. Apply the specified reduction (e.g., max or mean)
                Z[d, h_out, w_out] = reducer(window)

                # 2. Save necessary state (used ONLY by MaxPoolMethod for the mask)
                state_saver(d, h_out, w_out, window, (h_start, h_end, w_start, w_end))

    return Z


# --- Base Abstraction for Pooling Methods ---


class ReducerMethod:
    """Base class for Max and Average pooling methods."""

    def __init__(self, size: int, stride: int):
        self.size = size
        self.stride = stride
        self.X_shape: Optional[Tuple[int, int, int]] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Applies the pooling operation using the unified helper."""
        raise NotImplementedError

    def restore(self, grad_out: np.ndarray) -> np.ndarray:
        """Restores the gradient to the input shape (the backward pass)."""
        raise NotImplementedError


# --- Max Pooling Implementation ---


class MaxPoolMethod(ReducerMethod):
    """
    Implements Max Pooling. Forward pass is shared, but saves the max mask.
    Backward pass only passes gradient through the max-value locations.
    """

    def __init__(self, size: int, stride: int):
        super().__init__(size, stride)
        self.mask_indices: Dict = {}

    def _save_max_mask(
        self,
        d: int,
        h_out: int,
        w_out: int,
        window: np.ndarray,
        indices: Tuple[int, int, int, int],
    ):
        """
        Callback function used in the unified forward pass to save the mask indices.
        """
        h_start, _, w_start, _ = indices

        # Find the local index (h_local, w_local) of the max value
        h_local, w_local = np.unravel_index(np.argmax(window), window.shape)

        # Store the global index: (row_in, col_in)
        global_h = h_start + h_local
        global_w = w_start + w_local

        # Key for the dictionary is the output index (d, h_out, w_out)
        self.mask_indices[(d, h_out, w_out)] = (global_h, global_w)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Performs 2D Max Pooling using the unified _block_reduce_forward."""
        self.X_shape = X.shape
        self.mask_indices = {}  # Reset mask indices for new forward pass

        # Pass np.max as the reducer and _save_max_mask as the state saver
        Z = _block_reduce_forward(
            X, self.size, self.stride, np.max, self._save_max_mask
        )
        return Z

    def restore(self, grad_out: np.ndarray) -> np.ndarray:
        """Backward pass for Max Pooling."""
        if self.X_shape is None:
            raise ValueError("MaxPoolMethod restore called before forward pass.")

        D_in, H_in, W_in = self.X_shape
        dX = np.zeros(self.X_shape)

        # Iterate over the output indices (where the gradient is known)
        D_out, H_out, W_out = grad_out.shape

        for d in range(D_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    # Get the global indices of the max element from the forward pass
                    if (d, h_out, w_out) in self.mask_indices:
                        global_h, global_w = self.mask_indices[(d, h_out, w_out)]

                        # The incoming gradient is placed ONLY at the max location in dX.
                        dX[d, global_h, global_w] += grad_out[d, h_out, w_out]

        return dX


# --- Average Pooling Implementation ---


class AvgPoolMethod(ReducerMethod):
    """
    Implements Average Pooling. Forward pass is shared, but does not save specific state.
    Backward pass distributes the gradient uniformly.
    """

    def __init__(self, size: int, stride: int):
        super().__init__(size, stride)
        self.pool_area = size * size  # Size of the pooling window

    def _do_nothing(self, *args, **kwargs):
        """Placeholder for the state_saver callback, since AvgPool doesn't need to save state."""
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Performs 2D Average Pooling using the unified _block_reduce_forward."""
        self.X_shape = X.shape

        # Pass np.mean as the reducer and _do_nothing as the state saver
        Z = _block_reduce_forward(X, self.size, self.stride, np.mean, self._do_nothing)
        return Z

    def restore(self, grad_out: np.ndarray) -> np.ndarray:
        """Backward pass for Average Pooling."""
        if self.X_shape is None:
            raise ValueError("AvgPoolMethod restore called before forward pass.")

        D_in, H_in, W_in = self.X_shape
        S, K = self.stride, self.size

        dX = np.zeros(self.X_shape)

        # The fraction of the gradient each input pixel receives
        grad_contribution = 1.0 / self.pool_area

        D_out, H_out, W_out = grad_out.shape

        for d in range(D_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start, h_end = h_out * S, h_out * S + K
                    w_start, w_end = w_out * S, w_out * S + K

                    # Gradient is distributed uniformly
                    grad_to_distribute = grad_out[d, h_out, w_out] * grad_contribution

                    # Add the distributed gradient to the corresponding region in dX
                    dX[d, h_start:h_end, w_start:w_end] += grad_to_distribute

        return dX


# --- The Pool2D Layer Wrapper ---


class Pool2D:
    """
    A 2D Pooling layer that wraps a specific pooling method (Max or Avg).
    Input to this layer should be the 3D output of a Conv/Activation layer: (D, H, W).
    """

    def __init__(self, pool_type: Literal["max", "avg"], size: int, stride: int):
        if pool_type == "max":
            self.method: ReducerMethod = MaxPoolMethod(size, stride)
        elif pool_type == "avg":
            self.method: ReducerMethod = AvgPoolMethod(size, stride)
        else:
            raise ValueError("pool_type must be 'max' or 'avg'.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Passes the input to the selected pooling method."""
        if X.ndim != 3:
            raise ValueError(
                f"Input to Pool2D must be 3D (D, H, W), got {X.ndim} dimensions."
            )

        return self.method.forward(X)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Passes the gradient back to the selected pooling method for restoration."""
        return self.method.restore(grad_out)
