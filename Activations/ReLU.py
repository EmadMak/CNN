class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def derivative(self, grad_out):
        return grad_out * self.mask
