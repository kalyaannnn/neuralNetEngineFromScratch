import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tensor import Tensor
from src.optim import SGD, Adam, RMSprop

class TestOptimizers(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.param = Tensor([1.0, 2.0], requires_grad=True)
        self.grad = np.array([0.1, 0.2])

    def test_sgd(self):
        optimizer = SGD([self.param], lr=0.1)
        self.param.grad = self.grad.copy()
        optimizer.step()
        self.assertTrue(np.allclose(
            self.param.data,
            np.array([0.99, 1.98])
        ))

    def test_adam(self):
        optimizer = Adam([self.param], lr=0.1)
        self.param.grad = self.grad.copy()
        optimizer.step()
        # First step of Adam with default parameters
        self.assertTrue(np.all(self.param.data < np.array([1.0, 2.0])))

    def test_rmsprop(self):
        optimizer = RMSprop([self.param], lr=0.1)
        self.param.grad = self.grad.copy()
        optimizer.step()
        # First step of RMSprop with default parameters
        self.assertTrue(np.all(self.param.data < np.array([1.0, 2.0])))

if __name__ == '__main__':
    unittest.main()