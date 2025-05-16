import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tensor import Tensor

class TestTensor(unittest.TestCase):
    def test_basic_operations(self):
        # Test addition
        x = Tensor([1, 2, 3], requires_grad=True)
        y = Tensor([4, 5, 6], requires_grad=True)
        z = x + y
        z.backward()
        self.assertTrue(np.array_equal(x.grad, np.ones_like(x.data)))
        self.assertTrue(np.array_equal(y.grad, np.ones_like(y.data)))

        # Test multiplication
        x = Tensor([1, 2, 3], requires_grad=True)
        y = Tensor([4, 5, 6], requires_grad=True)
        z = x * y
        z.backward()
        self.assertTrue(np.array_equal(x.grad, y.data))
        self.assertTrue(np.array_equal(y.grad, x.data))

        # Test division
        x = Tensor([6, 8, 9], requires_grad=True)
        y = Tensor([2, 4, 3], requires_grad=True)
        z = x / y
        z.backward()
        self.assertTrue(np.array_equal(x.grad, 1/y.data))
        self.assertTrue(np.array_equal(y.grad, -x.data/(y.data**2)))

        # Test matrix multiplication
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = Tensor([[5, 6], [7, 8]], requires_grad=True)
        z = x.matmul(y)
        z.backward()
        self.assertTrue(z.data.shape == (2, 2))

    def test_reduction_operations(self):
        # Test sum
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.sum()
        y.backward()
        self.assertTrue(np.array_equal(x.grad, np.ones_like(x.data)))

        # Test mean
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.mean()
        y.backward()
        self.assertTrue(np.array_equal(x.grad, np.ones_like(x.data)/x.data.size))

        # Test max
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.max()
        y.backward()
        expected = np.zeros_like(x.data)
        expected[1, 1] = 1
        self.assertTrue(np.array_equal(x.grad, expected))

        # Test min
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.min()
        y.backward()
        expected = np.zeros_like(x.data)
        expected[0, 0] = 1
        self.assertTrue(np.array_equal(x.grad, expected))

    def test_activation_functions(self):
        # Test ReLU
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        y = x.relu()
        y.backward()
        expected = np.array([0, 0, 0, 1, 1])
        self.assertTrue(np.array_equal(x.grad, expected))

        # Test Sigmoid
        x = Tensor([0], requires_grad=True)
        y = x.sigmoid()
        y.backward()
        self.assertAlmostEqual(x.grad[0], 0.25, places=2)

        # Test Tanh
        x = Tensor([0], requires_grad=True)
        y = x.tanh()
        y.backward()
        self.assertAlmostEqual(x.grad[0], 1.0, places=2)

        # Test Softmax
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.softmax()
        y.backward()
        self.assertTrue(y.data.sum() - 1.0 < 1e-7)

    def test_shape_operations(self):
        # Test reshape
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.reshape(4)
        y.backward()
        self.assertTrue(np.array_equal(x.grad, np.ones_like(x.data)))

        # Test transpose
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.transpose()
        y.backward()
        self.assertTrue(np.array_equal(x.grad, np.ones_like(x.data)))

        # Test squeeze
        x = Tensor([[[1], [2]]], requires_grad=True)
        y = x.squeeze()
        y.backward()
        self.assertTrue(np.array_equal(x.grad.squeeze(), np.ones(2)))

        # Test unsqueeze
        x = Tensor([1, 2], requires_grad=True)
        y = x.unsqueeze(0)
        y.backward()
        self.assertTrue(np.array_equal(x.grad, np.ones_like(x.data)))

    def test_broadcasting(self):
        # Test broadcasting in addition
        x = Tensor([[1], [2]], requires_grad=True)  # (2,1)
        y = Tensor([1, 2, 3], requires_grad=True)   # (3,)
        z = x + y  # Should broadcast to (2,3)
        z.backward()
        self.assertEqual(x.grad.shape, x.data.shape)
        self.assertEqual(y.grad.shape, y.data.shape)

if __name__ == '__main__':
    unittest.main()