import numpy as np


class Pool2D:
    def __init__(self, pool_size=2, stride=1, mode="max"):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode.lower()

        if self.mode not in ["max", "average", "global_average", "global_max"]:
            raise ValueError(
                "Mode must be 'max', 'average', 'global_average', or 'global_max'."
            )

    def forward(self, input):
        # Input shape: (B, H, W, D)
        B, H, W, D = input.shape
        self.last_input = input

        # --- Handle Global Pooling ---
        if "global" in self.mode:
            # Output is (B, 1, 1, D)
            if self.mode == "global_average":
                output = np.mean(input, axis=(1, 2), keepdims=True)
            else:  # global_max
                output = np.max(input, axis=(1, 2), keepdims=True)
            self.mask = (H, W)  # Store H and W for backward pass scaling
            return output

        # --- Handle Local Pooling (e.g., 2x2, stride 2) ---

        # Calculate output dimensions
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        output = np.zeros((B, H_out, W_out, D))

        # Mask is only needed for Max Pooling
        if self.mode == "max":
            # Stores the indices of the max value (for backprop)
            self.mask = np.zeros(
                output.shape + (2,), dtype=np.int32
            )  # Shape: (B, H_out, W_out, D, 2)

        for b in range(B):
            for d in range(D):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride

                        region = input[
                            b,
                            h_start : h_start + self.pool_size,
                            w_start : w_start + self.pool_size,
                            d,
                        ]

                        if self.mode == "max":
                            output[b, i, j, d] = np.max(region)
                            # Store the location of the max value
                            max_loc = np.unravel_index(np.argmax(region), region.shape)
                            self.mask[b, i, j, d] = [
                                h_start + max_loc[0],
                                w_start + max_loc[1],
                            ]

                        elif self.mode == "average":
                            output[b, i, j, d] = np.mean(region)

        return output

    def backprop(self, d_L_d_out, learn_rate=None):
        # d_L_d_out shape: (B, H_out, W_out, D) or (B, 1, 1, D) for global pooling

        # d_L_d_input must have the shape of the original input: (B, H, W, D)
        d_L_d_input = np.zeros(self.last_input.shape)

        B, H, W, D = self.last_input.shape

        # --- Global Pooling Backward Pass ---
        if "global" in self.mode:
            H_in, W_in = self.mask  # Mask stores (H, W) for global pooling

            if self.mode == "global_average":
                H_in, W_in = self.last_input.shape[1:3]

                # incoming gradient shape: (B, 1, 1, D)
                # We want output: (B, H_in, W_in, D)

                # Spread gradient evenly across all pixels
                d_L_d_input = np.ones_like(self.last_input) * (
                    d_L_d_out / (H_in * W_in)
                )
            else:  # global_max
                # Global Max needs to create a mask on the fly because we didn't store one
                for b in range(B):
                    for d in range(D):
                        # Find the index of the max value across the H x W feature map
                        max_loc = np.unravel_index(
                            np.argmax(self.last_input[b, :, :, d]), (H, W)
                        )
                        # Place the incoming gradient dL/dA only at the max location
                        d_L_d_input[b, max_loc[0], max_loc[1], d] = d_L_d_out[
                            b, 0, 0, d
                        ]

            return d_L_d_input

        # --- Local Pooling Backward Pass ---

        H_out, W_out = d_L_d_out.shape[1:3]

        for b in range(B):
            for d in range(D):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride

                        # The incoming gradient element for this region
                        grad_element = d_L_d_out[b, i, j, d]

                        # if self.mode == "max":
                        #     # Max Pooling: Gradient only goes to the max index
                        #     h_idx, w_idx = self.mask[b, i, j, d].astype(int)
                        #     d_L_d_input[b, h_idx, w_idx, d] += grad_element

                        # FIX: Explicitly index the stored H and W coordinates from the 5D mask
                        if self.mode == "max":
                            grad_element = d_L_d_out[b, i, j, d]

                            # h_idx is stored at index 0 of the last dimension, w_idx is at index 1
                            h_idx = self.mask[b, i, j, d, 0]
                            w_idx = self.mask[b, i, j, d, 1]

                            # Assign the gradient to the correct location (using .astype(int) is no longer needed)
                            d_L_d_input[b, h_idx, w_idx, d] += grad_element

                        elif self.mode == "average":
                            # Average Pooling: Gradient is distributed equally
                            scaling_factor = self.pool_size * self.pool_size

                            # Distribute the scaled gradient element across the region
                            d_L_d_input[
                                b,
                                h_start : h_start + self.pool_size,
                                w_start : w_start + self.pool_size,
                                d,
                            ] += grad_element / scaling_factor
        return d_L_d_input
