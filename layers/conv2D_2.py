import numpy as np
from Activations import ReLU, SiLU


class Conv2D:
    # A Convolution layer using 3x3 filters for multi-channel input (H, W, D).

    def __init__(
        self, kernel_number, input_depth, activation=ReLU(), padding_mode: str = "valid"
    ):
        self.kernel_number = kernel_number
        self.input_depth = input_depth
        self.activation = activation

        self.padding_mode = padding_mode.lower()
        if self.padding_mode not in ["valid", "same"]:
            raise ValueError("padding_mode must be 'valid' or 'same'.")

        # Filters are 4D: (num_filters, 3, 3, input_depth)
        # Filters must span the full depth (D) of the input.
        self.filters = self.initialize_kernels(self.kernel_number, 3, self.input_depth)
        self.biases = np.zeros((1, 1, 1, kernel_number))

    def initialize_kernels(
        self, kernel_number: int, kernel_size: int, input_channels: int = 1
    ):
        # Calculate Fan-in (n_in): kernel_height * kernel_width * input_channels
        fan_in = kernel_size * kernel_size * input_channels

        # Calculate the standard deviation for the scaling factor
        std_dev = np.sqrt(2.0 / fan_in)

        # Initialize kernels from a standard normal distribution and scale them
        # Shape: (C_out, K, K)
        kernels = (
            np.random.randn(kernel_number, kernel_size, kernel_size, input_channels)
            * std_dev
        )

        return kernels

    def iterate_regions(self, image):
        """
        Generates all possible 3x3 image regions using valid padding.
        - image is a 3D numpy array: (H, W, D).
        """
        # The shape now includes the depth (D)
        h, w, d = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                # im_region is a 3D slice: (3, 3, D)
                im_region = image[i : (i + 3), j : (j + 3), :]
                yield im_region, i, j

    def forward(self, input):
        """
        Performs a forward pass.
        Returns a 3D numpy array with dimensions (H', W', num_filters).
        - input is a 4D numpy array (B, H, W, D).
        """
        self.last_input = input

        # Check to ensure input depth matches filter depth
        if input.shape[3] != self.input_depth:
            raise ValueError("Input depth does not match filter depth.")

        # Apply padding
        # For 3x3: 1 if 'same' padding, and 0 if 'valid' padding
        pad_amount = 1 if self.padding_mode == "same" else 0

        # Apply padding if required
        self.padded_input = np.pad(
            input,
            # Pad 'pad amount' on height and width axis, no padding on depth or batch
            ((0, 0), (pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        B, H, W, D = input.shape

        # Calculate output dimensions based on padding mode
        output_h = H if self.padding_mode == "same" else H - 2
        output_w = W if self.padding_mode == "same" else W - 2

        # Output is (H-2, W-2, num_filters)
        output_raw = np.zeros((B, output_h, output_w, self.kernel_number))

        for b in range(B):
            # Iterate regions over the padded input for this specific image 'b'
            # self.padded_input[b] has shape (H_p, W_p, D)
            for im_region, i, j in self.iterate_regions(self.padded_input[b]):
                # Convolution: Element-wise multiply the (3, 3, D) region by the (Nf, 3, 3, D) filters,
                # then sum the result across all three inner axes (1, 2, 3) for each filter.
                # This implements the Dot Product (Volume Correlation) for each filter.
                output_raw[b, i, j] = np.sum(im_region * self.filters, axis=(1, 2, 3))

        output_biased = output_raw + self.biases
        self.last_output_raw = output_biased

        # use an activation function on the biased output so biases influence activations
        output_activated = self.activation.forward(output_biased)
        return output_activated

    def backprop(self, d_L_d_out, learn_rate):
        """
        Performs a backward pass. Calculates d_L_d_filters and d_L_d_input.
        - d_L_d_out is the loss gradient for this layer's outputs (H', W', Nf).
        - learn_rate is a float.
        Returns the loss gradient for this layer's inputs (d_L_d_input), shape (H, W, D).
        """
        # Calculate gradient with respect to filters (d_L_d_filters)
        # d_L_d_filters must be the same shape as self.filters: (Nf, 3, 3, D)
        d_L_d_filters = np.zeros(self.filters.shape)

        # d_L_d_out is now dL/dA (Loss gradient w.r.t activated output)
        # d_L_d_conv_out is now dL/dZ (Loss gradient w.r.t raw convolution output (aka before activation))
        d_L_d_conv_out = self.activation.derivative(d_L_d_out)

        # Calculate gradient for biases
        d_L_d_B = np.sum(d_L_d_conv_out, axis=(0, 1, 2), keepdims=True)

        # d_L_d_input must be the same shape as the input: (H, W, D)
        b_pad, h_pad, w_pad, d_pad = self.last_input.shape

        # d_L_d_input_padded must be the size of the padded input used in forward.
        pad_amount = 1 if self.padding_mode == "same" else 0
        d_L_d_input_padded = np.zeros(self.padded_input.shape)

        B, output_h, output_w, _ = d_L_d_out.shape

        for b in range(B):
            for i in range(output_h):
                for j in range(output_w):
                    # Region used in forward: self.padded_input[i:i+3, j:j+3, :]
                    im_region = self.padded_input[b, i : (i + 3), j : (j + 3), :]

                    for f in range(self.kernel_number):
                        scalar_grad = d_L_d_conv_out[b, i, j, f]

                        # Accumulate dL/dF = dL/dOut * Input_Region
                        d_L_d_filters[f] += scalar_grad * im_region

                        # Accumulate dL/dIn = dL/dOut * Filter
                        # Place gradient into the padded input gradient array
                        d_L_d_input_padded[b, i : (i + 3), j : (j + 3), :] += (
                            scalar_grad * self.filters[f]
                        )

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # Update biases
        self.biases -= learn_rate * d_L_d_B

        if self.padding_mode == "same":
            # Crop the border (1 pixel on all sides)
            d_L_d_input = d_L_d_input_padded[:, 1:-1, 1:-1, :]
        else:  # 'valid' mode, no padding was added
            d_L_d_input = d_L_d_input_padded

        # Return the gradient with respect to the input
        return d_L_d_input
