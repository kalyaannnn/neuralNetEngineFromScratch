import numpy as np
from typing import List, Dict
from .tensor import Tensor


class Optimizer:
    def __init__(self, parameters : List[Tensor]):
        self.parameters = list(parameters)

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, parameters : List[Tensor], lr : float = 0.01, momentum : float = 0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] + param.grad
                param.data -= self.lr * self.velocity[i]
            else:
                param.data -= self.lr * param.grad


class Adam(Optimizer):
    def __init__(self, parameters : List[Tensor], lr : float = 0.001, betas : tuple = (0.9, 0.999), eps : float = 1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0


    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad**2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)  

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)



class RMSProp(Optimizer):
    def __init__(self, parameters : List[Tensor], lr : float = 0.01, alpha : float = 0.99,
                  eps : float = 1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_avg = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            self.square_avg[i] = self.alpha * self.square_avg[i] + \
            (1 - self.alpha) * np.square(param.grad)

            param.data -= self.lr * param.grad / (np.sqrt(self.square_avg[i]) + self.eps)

    