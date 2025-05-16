import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tensor import Tensor
from src.layers import Linear, ReLU, Softmax
from src.loss import CrossEntropyLoss

class TestLayers(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_linear_layer(self):
        layer = Linear(2, 3)
        self.assertEqual(layer.weight.data.shape, (2, 3))
        if layer.bias is not None:
            self.assertEqual(layer.bias.data.shape, (3,))

        x = Tensor([[1.0, 2.0]], requires_grad=True)
        out = layer(x)
        self.assertEqual(out.data.shape, (1, 3))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)
        # Check gradient shapes
        self.assertEqual(layer.weight.grad.shape, (2, 3))
        self.assertEqual(layer.bias.grad.shape, (3,))

    def test_relu_layer(self):
        relu = ReLU()
        x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        out = relu(x)
        self.assertTrue(np.array_equal(
            out.data,
            np.array([0.0, 0.0, 1.0])
        ))

        out.backward()
        self.assertTrue(np.array_equal(
            x.grad,
            np.array([0.0, 0.0, 1.0])
        ))

    def test_softmax_layer(self):
        softmax = Softmax()
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        out = softmax(x)
        
        self.assertAlmostEqual(float(out.data.sum()), 1.0, places=5)
        
        # Compute loss as the first element of the output
        loss = out[0, 0]
        loss.backward()
        
        # Calculate expected gradients
        s = np.exp([1.0, 2.0, 3.0])
        s /= s.sum()
        expected_grad = np.array([
            s[0] * (1 - s[0]),
            -s[0] * s[1],
            -s[0] * s[2]
        ])
        
        self.assertTrue(np.allclose(x.grad[0], expected_grad, atol=1e-6))

    def test_cross_entropy_loss(self):
        criterion = CrossEntropyLoss()
        # Use raw logits instead of probabilities
        predicted = Tensor([[1.0, 3.0, 2.0]], requires_grad=True)
        target = Tensor([1])
        
        loss = criterion(predicted, target)
        loss.backward()
        
        self.assertIsNotNone(predicted.grad)
        
        # Expected loss calculation
        logits = np.array([1.0, 3.0, 2.0])
        exp_logits = np.exp(logits - np.max(logits))
        softmax = exp_logits / exp_logits.sum()
        expected_loss = -np.log(softmax[1])
        self.assertAlmostEqual(loss.data, expected_loss, places=5)

if __name__ == '__main__':
    unittest.main()