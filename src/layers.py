import numpy as np
from .tensor import Tensor
from .model import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        # Initialize weights using Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(
                np.zeros(out_features),
                requires_grad=True
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
    def parameters(self):
        return []
    
class Softmax(Module):
    def __init__(self, axis = -1):
        self.axis = axis
    
    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(self.axis)
    
    def parameters(self):
        return []

class CrossEntropyLoss:
    def __call__(self, pred, target):
        # Ensure inputs are 2D
        if len(pred.data.shape) == 1:
            pred = pred.unsqueeze(0)
        if len(target.data.shape) == 0:
            target = target.unsqueeze(0)
            
        # Convert target to one-hot encoding
        batch_size = pred.data.shape[0]
        num_classes = pred.data.shape[1]
        target_one_hot = np.zeros((batch_size, num_classes))
        target_one_hot[np.arange(batch_size), target.data] = 1
        
        # Compute softmax
        exp_pred = np.exp(pred.data - np.max(pred.data, axis=1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # Compute loss
        loss = -np.sum(target_one_hot * np.log(softmax_pred + 1e-10)) / batch_size
        
        # Create tensor for loss
        loss_tensor = Tensor(loss, requires_grad=True)
        
        # Define backward pass
        def backward(grad):
            # Gradient of cross entropy loss with respect to predictions
            grad_pred = (softmax_pred - target_one_hot) / batch_size
            pred.backward(grad * grad_pred)
            
        loss_tensor._backward = backward
        return loss_tensor

def grad_fn(self, grad):
    """Compute gradients for this operation"""
    if self.requires_grad:
        # Initialize self_grad
        self_grad = grad
        
        # Handle broadcasting if needed
        if self.data.shape != grad.shape:
            self_grad = self._unbroadcast(self_grad, self.data.shape)
        
        # Accumulate gradients
        if self.grad is None:
            self.grad = self_grad
        else:
            self.grad += self_grad
        
        # Propagate gradients to input tensors
        if hasattr(self, '_backward') and self._backward is not None:
            self._backward(grad)