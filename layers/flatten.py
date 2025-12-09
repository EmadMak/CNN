class Flatten:
    def forward(self, input):
        """
        Converts 4D input (B, H, W, D) to 2D output (B, H*W*D).
        """
        self.original_shape = input.shape  # Store the original 4D shape
        B = input.shape[0]

        # Reshape to (Batch Size, -1) where -1 automatically calculates H*W*D
        flattened_output = input.reshape(B, -1)
        return flattened_output

    def backprop(self, d_L_d_out, learn_rate=None):
        """
        Reshapes the incoming 2D gradient back to the original 4D shape.
        Learn rate is ignored as there are no parameters to update.
        """
        # Reshape the incoming 2D gradient back to the 4D shape stored in forward pass
        d_L_d_input = d_L_d_out.reshape(self.original_shape)

        return d_L_d_input
