from typing import Tuple

import numpy as np

# Tried simulating scikit images library
from scikit_clone.measure import ArrayReducer, block_reduce


# Keep input as numpy array
class Activations:
    class ReLU:
        def __init__(self):
            self.mask = None

        def forward(self, x):
            self.mask = x > 0
            return x * self.mask

        def derivative(self, grad_out):
            return grad_out * self.mask

    class SiLU:
        def __init__(self):
            self.x = None
            self.sigma_x = None

        def forward(self, x):
            self.x = x
            self.sigma_x = 1 / (1 + np.exp(-x))
            return x * self.sigma_x

        def derivative(self, grad_out):
            if self.x is None or self.sigma_x is None:
                raise ValueError(
                    "SiLU derivative called before forward pass. State (self.x, self.sigma_x) is None."
                )
            grad_in = self.sigma_x * (1 + self.x * (1 - self.sigma_x))
            return grad_out * grad_in

    class GELU:
        def __init__(self):
            self.x = None
            self.u = None
            self.tanh_u = None

        def forward(self, x: np.ndarray):
            self.x = x
            self.u = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
            self.tanh_u = np.tanh(self.u)
            return 0.5 * x * (1 + self.tanh_u)

        def derivative(self, grad_out):
            if self.x is None or self.tanh_u is None:
                raise ValueError(
                    "GELU derivative called before forward pass. State (self.x, self.tanh_u) is None."
                )

            du_dx = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.x**2)
            grad = 0.5 * (1 + self.tanh_u) + 0.5 * self.x * (1 - self.tanh_u**2) * du_dx
            return grad_out * grad

    class MeanPool:
        def __init__(self):
            self.block_size = None
            self.block_area = 1
            self.reducer = np.mean

        def forward(self, X: np.ndarray, block_size: Tuple[int, int, int]):
            self.block_size = block_size

            _, H_block, W_block = block_size
            self.block_area = H_block * W_block

            return block_reduce(X=X, filter_shape=block_size, reducer=self.reducer)

        def restore(self, dZ: np.ndarray, input_shape: Tuple[int, int, int]):
            D, H_in, W_in = input_shape
            D_out, H_out, W_out = dZ.shape

            # Calculate the size of the pooling block (e.g., 2 for 2x2 pooling)
            H_block = H_in // H_out
            W_block = W_in // W_out

            if H_block * W_block != self.block_area:
                # Sanity check to ensure block_area was correctly stored in forward pass
                raise ValueError(
                    "Block size mismatch in restore. Check forward pass calculation."
                )

            # Initialize the input gradient array (dX)
            dX = np.zeros(input_shape, dtype=dZ.dtype)

            # Calculate the gradient to be scattered into each pixel (dZ / Area)
            gradient_per_pixel = dZ / self.block_area

            # Iterate over the output coordinates and scatter the gradient
            for d in range(D):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start, w_start = h * H_block, w * W_block
                        h_end, w_end = h_start + H_block, w_start + W_block
                        # Distribute the single gradient_per_pixel value across the H_block x W_block area
                        dX[d, h_start:h_end, w_start:w_end] = gradient_per_pixel[
                            d, h, w
                        ]
            return dX

    # class MaxPool:
    #     def __init__(self):
    #     self.mask_indices = None
    #     self.reducer: ArrayReducer = np.max

    #     def forward(self, X: np.ndarray, block_size: Tuple[int, int, int]):
    #         Z = block_reduce(X=X, filter_shape=block_size, reducer=self.reducer)
    #         return Z

    #     def restore(self, dZ:np.ndarray, input_shape: Tuple[int, int, int]):

    class MaxPool:
        """
        Max Pooling implementation. This class performs the forward pass directly
        (instead of using block_reduce) in order to calculate and store the mask
        of maximum indices required for the backward pass.
        """

        def __init__(self):
            # mask_indices will store the (h, w) coordinates of the max value for each block
            self.mask_indices = {}
            self.reducer: ArrayReducer = np.max

        def forward(
            self, X: np.ndarray, block_size: Tuple[int, int, int]
        ) -> np.ndarray:
            """
            Applies Max Pooling to the input X, capturing the indices of the max values.

            """
            self.mask_indices = {}

            _, H_block, W_block = block_size
            D_in, H_in, W_in = X.shape

            if H_in % H_block != 0 or W_in % W_block != 0:
                raise ValueError(
                    "Input dimensions must be divisible by the block size for simple pooling."
                )

            H_out = H_in // H_block
            W_out = W_in // W_block

            Z = np.zeros((D_in, H_out, W_out), dtype=X.dtype)

            # Iterate through channels, output height, and output width
            for d in range(D_in):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start, w_start = h * H_block, w * W_block
                        h_end, w_end = h_start + H_block, w_start + W_block

                        # Extract the current block
                        block = X[d, h_start:h_end, w_start:w_end]

                        # Compute the max value
                        max_val = np.max(block)
                        Z[d, h, w] = max_val

                        # Find the location (index) of the max value within the block
                        # np.argmax flattens the block, so we need to convert the flat index back to 2D
                        flat_index = np.argmax(block)
                        h_idx, w_idx = np.unravel_index(flat_index, block.shape)

                        # Store the absolute coordinates in the original input X
                        abs_h = h_start + h_idx
                        abs_w = w_start + w_idx

                        # Store the mask coordinates: (channel, output_h, output_w) -> (input_h, input_w)
                        self.mask_indices[(d, h, w)] = (abs_h, abs_w)

            return Z

        def restore(
            self, dZ: np.ndarray, input_shape: Tuple[int, int, int]
        ) -> np.ndarray:
            """
            Backward pass for Max Pooling (Unpooling).
            The gradient dZ is routed only to the single pixel location that
            held the maximum value during the forward pass (using mask_indices).
            """
            if not self.mask_indices:
                raise RuntimeError(
                    "Forward pass must be run before backward pass for MaxPoolMethod to create the mask."
                )

            dX = np.zeros(input_shape, dtype=dZ.dtype)
            D_out, H_out, W_out = dZ.shape

            # Iterate over the output gradient (dZ)
            for d in range(D_out):
                for h in range(H_out):
                    for w in range(W_out):
                        # Get the absolute (h, w) coordinate in the input volume (X)
                        # where the max value was located
                        abs_h, abs_w = self.mask_indices[(d, h, w)]

                        # Scatter the incoming gradient dZ[d, h, w] to the specific max location in dX
                        dX[d, abs_h, abs_w] = dZ[d, h, w]

            return dX
