from typing import Callable, Tuple

import numpy as np

ArrayReducer = Callable[[np.ndarray], float]


def zero_pad(X: np.ndarray, pad: int) -> np.ndarray:
    """Helper function for applying zero padding to the spatial dimensions of a 3D volume (D, H, W)."""
    if len(X.shape) == 2:
        print(
            "Warning: Input array is 2D. Something is overlooked in the convolutional layer?"
        )
        return np.pad(X, ((pad, pad), (pad, pad)), "constant", constant_values=(0))

    return np.pad(X, ((0, 0), (pad, pad), (pad, pad)), "constant", constant_values=(0))


#
# def correlate2d(
#    X: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1
# ) -> np.ndarray:
#    # Apply zero padding
#    X_padded = zero_pad(X, padding)
#
#    H_out = (X_padded.shape[0] - kernel.shape[0]) // stride + 1
#    W_out = (X_padded.shape[1] - kernel.shape[1]) // stride + 1
#
#    output = np.zeros((H_out, W_out))
#
#    for i in range(0, H_out):
#        for j in range(0, W_out):
#            region = X_padded[
#                i * stride : i * stride + kernel.shape[0],
#                j * stride : j * stride + kernel.shape[1],
#            ]
#            output[i, j] = np.sum(region * kernel)
#
#    return output
#
#
# def convolve2d(X: np.ndarray, kernels: np.ndarray, padding: int = 0, stride: int = 1):
#    """
#    X: H x W  (grayscale input)
#    kernels: num_filters x kH x kW
#    Returns: H_out x W_out x num_filters
#    """
#    num_filters = kernels.shape[0]
#    H_out = (X.shape[0] + 2 * padding - kernels.shape[1]) // stride + 1
#    W_out = (X.shape[1] + 2 * padding - kernels.shape[2]) // stride + 1
#
#    output = np.zeros((H_out, W_out, num_filters))
#
#    for f in range(num_filters):
#        output[:, :, f] = correlate2d(X, kernels[f], padding, stride)
#
#    return output
#
#


def correlate2d(
    X: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1
) -> np.ndarray:
    """
    Performs multi-channel 2D correlation.

    Args:
        X: np.ndarray of shape (C_in, H, W) or (H, W) for single-channel.
        kernel: np.ndarray of shape (C_in, kH, kW) or (kH, kW) for single-channel.
        padding: int, amount of zero-padding.
        stride: int, stride for sliding the kernel.

    Returns:
        np.ndarray: 2D output (H_out, W_out)
    """

    C_in, H_in, W_in = X.shape
    print("kernel shape: ", kernel.shape)
    _, kH, kW = kernel.shape

    # Pad input
    X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding)))

    H_padded, W_padded = X_padded.shape[1], X_padded.shape[2]

    # Output size
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start, h_end = i * stride, i * stride + kH
            w_start, w_end = j * stride, j * stride + kW

            region = X_padded[:, h_start:h_end, w_start:w_end]  # shape: (C_in, kH, kW)
            output[i, j] = np.sum(region * kernel)  # sum over all channels

    return output


def convolve2d_multi_filter(
    X: np.ndarray, kernels: np.ndarray, padding: int = 0, stride: int = 1
):
    """
    Wrapper function to apply multiple 2D filters (kernels) to a single channel input (X).

    X: H x W (grayscale input)
    kernels: num_filters x kH x kW
    Returns: H_out x W_out x num_filters
    """
    if kernels.ndim != 3:
        raise ValueError(
            "convolve2d_multi_filter requires kernels of shape (num_filters, kH, kW)."
        )

    num_filters = kernels.shape[0]

    # *** Note: This dimension calculation is correct ONLY if kernels is 3D ***
    kH = kernels.shape[1]
    kW = kernels.shape[2]

    H_in, W_in = X.shape
    H_out = (H_in + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1

    output = np.zeros((H_out, W_out, num_filters))

    for f in range(num_filters):
        # Pass the 2D kernel slice (kernels[f]) to the 2D correlation function
        output[:, :, f] = correlate2d(X, kernels[f], padding, stride)

    return output


def block_reduce(
    X: np.ndarray, filter_shape: Tuple[int, int, int], reducer: ArrayReducer
):
    if not callable(reducer):
        raise TypeError(
            "The 'reducer' argument must be a callable function (e.g., np.mean or np.max)."
        )

    D_block, H_block, W_block = filter_shape
    D_in, H_in, W_in = X.shape

    H_out = H_in // H_block
    W_out = W_in // W_block

    if D_block != 1:
        raise ValueError(
            "block_size depth dimension must be 1. Cannot reduce across channels."
        )

    if H_in % H_block != 0 or W_in % W_block != 0:
        raise ValueError(
            "Input dimensions must be divisible by the block size for simple pooling."
        )

    pooled_X = np.zeros((D_in, H_out, W_out), dtype=X.dtype)

    for d in range(D_in):
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * H_block
                h_end = h_start + H_block
                w_start = w * W_block
                w_end = w_start + W_block

                block = X[d, h_start:h_end, w_start:w_end]

                pooled_X[d, h, w] = reducer(block)

    return pooled_X
