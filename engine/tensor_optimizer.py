import numpy as np


class Optimizer:
    """
    Base optimizer class. Implements zero_grad and step interface.
    """
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        """Reset gradients for all parameters to zero."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        """Update parameters. To be implemented by subclasses."""
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Args:
        parameters: iterable of Tensors to update
        lr (float): learning rate
        momentum (float): momentum coefficient (0 = no momentum)
    """
    def __init__(self, parameters, lr=1e-3, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Perform a single optimization step."""
        for i, p in enumerate(self.parameters):
            # update velocity
            self.velocities[i] = self.momentum * self.velocities[i] + p.grad
            # apply parameter update
            p.data -= self.lr * self.velocities[i]

class Adam(Optimizer):
    """
    Adam optimizer.

    Args:
        parameters: iterable of Tensors to update
        lr (float): learning rate
        betas (tuple): coefficients for running averages (beta1, beta2)
        eps (float): term for numerical stability
    """
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Perform a single optimization step with bias correction."""
        self.t += 1
        for i, p in enumerate(self.parameters):
            g = p.grad
            # update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            # bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            # parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
