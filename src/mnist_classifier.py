from .model import Module
from .tensor import Tensor
from .layers import Linear, ReLU
import numpy as np

class MNISTClassifier(Module):
    def __init__(self):
        super().__init__()  
        
        # Define network layers
        self.fc1 = Linear(784, 128)
        self.relu1 = ReLU()
        self.fc2 = Linear(128, 64)
        self.relu2 = ReLU()
        self.fc3 = Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        # Flatten input: (batch_size, 1, 28, 28) -> (batch_size, 784)
        batch_size = x.data.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Forward pass through layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x