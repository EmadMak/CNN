import numpy as np


class SiLU:
    def __init__(self):
        self.x = None
        self.sigma_x = None

    def forward(self, x):
        self.x = x
        self.sigma_x = 1 / (1 + np.exp(-x))
        return x * self.sigma_x

    def derivative(self, grad_out):
        # Input d_A_d_Z
        if self.x is None or self.sigma_x is None:
            raise ValueError(
                "SiLU derivative called before forward pass. State (self.x, self.sigma_x) is None."
            )
        d_L_d_A = self.sigma_x * (1 + self.x * (1 - self.sigma_x))

        # return d_L_d_Z
        return grad_out * d_L_d_A
