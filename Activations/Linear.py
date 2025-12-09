class Linear:
    """Identity activation used for output logits."""

    def forward(self, x):
        return x

    def derivative(self, grad_out):
        return grad_out
