import numpy as np
from .tensor import Tensor

class CrossEntropyLoss:
    def __call__(self, predicted, target):
        # Compute softmax with numerical stability
        max_vals = predicted.data.max(axis=1, keepdims=True)
        exp_pred = np.exp(predicted.data - max_vals)
        softmax_pred = exp_pred / exp_pred.sum(axis=1, keepdims=True)
        
        # Get prediction for correct classes
        batch_size = predicted.data.shape[0]
        indices = np.arange(batch_size)
        correct_class_probs = softmax_pred[indices, target.data]
        
        # Compute negative log likelihood as a scalar
        loss_scalar = float(-np.log(correct_class_probs).sum() / batch_size)
        # Create tensor with scalar value
        loss = Tensor(loss_scalar, requires_grad=predicted.requires_grad)
        
        def grad_fn(grad):
            if predicted.requires_grad:
                # Compute gradient of cross entropy with respect to inputs
                grad_pred = softmax_pred.copy()
                grad_pred[indices, target.data] -= 1
                grad_pred /= batch_size
                predicted.grad += grad_pred * grad

        if loss.requires_grad:
            loss.grad_fn = grad_fn
            loss.children = [predicted]
            
        return loss

