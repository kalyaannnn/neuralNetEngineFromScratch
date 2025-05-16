import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mnist_classifier import MNISTClassifier
from src.tensor import Tensor

class TestMNISTClassifier(unittest.TestCase):
    def setUp(self):
        """Initialize the model before each test"""
        self.model = MNISTClassifier()

    def test_forward_pass(self):
        """Test the forward pass of the model"""
        # Create dummy batch of MNIST images
        batch_size = 2
        x = Tensor(np.random.randn(batch_size, 1, 28, 28))

        # Perform forward pass
        output = self.model(x)

        # Check output shape
        self.assertEqual(output.data.shape, (batch_size, 10))
    
    def test_parameter_count(self):
        """Test that the model has the correct number of parameters"""
        params = self.model.parameters()

        # Calculate total parameters
        total_params = sum(p.data.size for p in params)
        
        # Expected parameters:
        # fc1: 784 * 128 + 128 (weights + bias)
        # fc2: 128 * 64 + 64
        # fc3: 64 * 10 + 10
        expected_params = (784 * 128 + 128) + (128 * 64 + 64) + (64 * 10 + 10)
        self.assertEqual(total_params, expected_params)

    def test_train_eval_modes(self):
        """Test training and evaluation mode switching"""
        # Test train mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Test eval mode
        self.model.eval()
        self.assertFalse(self.model.training)

if __name__ == '__main__':
    unittest.main()