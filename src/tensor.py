import numpy as np

class Tensor:
    def __init__(self, data, requires_grad = False):

        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.children = []

        if requires_grad:
            self.zero_grad()
    
    def zero_grad(self):
        # Change to float64 type
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def backward(self, grad = None):
        topo_order = self._build_topological_order()

        # Change to float64 type
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for tensor in reversed(topo_order):
            if tensor.grad_fn:
                tensor.grad_fn(tensor.grad)

        if grad is not None:
            self.grad = grad
        
    def _build_topological_order(self):
        visited = set()
        topo_order = []

        def dfs(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor.children:
                    dfs(child)
                topo_order.append(tensor)
        
        dfs(self)
        return topo_order
    

    def __add__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data + other.data, requires_grad = self.requires_grad or other.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = grad
                if self.data.shape != out.data.shape:
                    self_grad = self._unbroadcast(grad, self.data.shape)
                self.grad += self_grad

            if other.requires_grad:
                other_grad = grad
                if other.data.shape != out.data.shape:
                    other_grad = self._unbroadcast(grad, other.data.shape)
                other.grad += other_grad
        
        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self, other]
        return out
    

    def _ensure_tensor(self, obj):
        if not isinstance(obj, Tensor):
            obj = Tensor(obj)
        return obj
    
    def _unbroadcast(self, grad, original_shape):
        # Ensure grad is float64
        grad = grad.astype(np.float64)
        
        # Sum across broadcast dimensions
        while len(grad.shape) > len(original_shape):
            grad = grad.sum(axis=0)
        
        for i, (orig, curr) in enumerate(zip(original_shape, grad.shape)):
            if orig == 1 and curr != 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        # Ensure output shape matches original shape
        return np.broadcast_to(grad, original_shape)
    


    def __mul__(self, other):
        other = self._ensure_tensor(other) 
        out = Tensor(self.data * other.data, requires_grad = self.requires_grad or other.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = grad * other.data
                if self.data.shape != out.data.shape:
                    self_grad = self._unbroadcast(self_grad, self.data.shape)
                self.grad += self_grad

            if other.requires_grad:
                other_grad = grad * self.data
                if other.data.shape != out.data.shape:
                    other_grad = self._unbroadcast(other_grad, other.data.shape)
                other.grad += other_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self, other]

        return out
    

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data - other.data, requires_grad = self.requires_grad or other.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = grad
                if self.data.shape != out.data.shape:
                    self_grad = self._unbroadcast(grad, self.data.shape)
                self.grad += self_grad

            if other.requires_grad:
                other_grad = -grad
                if other.data.shape != out.data.shape:
                    other_grad = self._unbroadcast(other_grad, other.data.shape)
                other.grad += other_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self, other]

        return out
    
    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data / other.data, requires_grad = self.requires_grad or other.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = grad / other.data
                if self.data.shape != out.data.shape:
                    self_grad = self._unbroadcast(self_grad, self.data.shape)
                self.grad += self_grad

            if other.requires_grad:
                other_grad = grad * (-self.data / (other.data * other.data))
                if other.data.shape != out.data.shape:
                    other_grad = self._unbroadcast(other_grad, other.data.shape)
                other.grad += other_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self, other]
        
        return out
    
    def matmul(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(np.matmul(self.data, other.data), requires_grad = self.requires_grad or other.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = np.matmul(grad, other.data.T)
                if self.data.shape != out.data.shape:
                    self_grad = self._unbroadcast(self_grad, self.data.shape)
                self.grad += self_grad

            if other.requires_grad:
                other_grad = np.matmul(self.data.T, grad)
                if other.data.shape != out.data.shape:
                    other_grad = self._unbroadcast(other_grad, other.data.shape)
                other.grad += other_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self, other]

        return out


    def sum(self, axis = None, keepdims = False):
        out = Tensor(self.data.sum(axis = axis, keepdims = keepdims), requires_grad = self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad: 
                if not keepdims and axis is not None:
                    shape = list(self.data.shape)
                    if isinstance(axis, (list, tuple)):
                        for ax in sorted(axis):
                            shape.insert(ax, 1)
                    else:
                        shape.insert(axis, 1)
                    grad = grad.reshape(shape)

                self_grad = grad * np.ones_like(self.data)
                self.grad += self_grad
            
        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]
            
        return out
    

    def mean(self, axis = None, keepdims = False):
        out = Tensor(self.data.mean(axis = axis, keepdims = keepdims), requires_grad = self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                if not keepdims and axis is not None:
                    shape = list(self.data.shape)
                    if isinstance(axis, (list, tuple)):
                        for ax in sorted(axis):
                            shape.insert(ax, 1)
                    else:
                        shape.insert(ax, 1)
                    grad = grad.reshape(shape)
                
                if axis is not None:
                   n = np.prod([self.data.shape[i] for i in 
                           (axis if isinstance(axis, (list, tuple)) else [axis])])
                else:
                    n = self.data.size
                self_grad = grad * np.ones_like(self.data) / n
                self.grad += self_grad
            
        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]
        
        return out
    
    def max(self, axis = None, keepdims = False):
        out = Tensor(self.data.max(axis = axis, keepdims = keepdims), requires_grad = self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                if not keepdims and axis is not None:
                    shape = list(self.data.shape)
                    if isinstance(axis, (list, tuple)):
                        for ax in sorted(axis):
                            shape.insert(ax, 1)
                    else:
                        shape.insert(ax, 1)
                    grad = grad.reshape(shape)

                mask = (self.data == np.max(self.data, axis = axis, keepdims = True))
                self_grad = grad * mask
                self.grad += self_grad
        
        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out
    
    def min(self, axis = None, keepdims = False):
        out = Tensor(self.data.min(axis = axis, keepdims = keepdims), requires_grad = self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                if not keepdims and axis is not None:
                    shape = list(self.data.shape)
                    if isinstance(axis, (list, tuple)):
                        for ax in sorted(axis):
                            shape.insert (ax, 1)
                    else:
                        shape.insert(ax, 1)
                    grad = grad.reshape(shape)

                mask = (self.data == np.min(self.data, axis = axis, keepdims = True))
                self_grad = grad * mask
                self.grad += self_grad
        
        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = (grad * (self.data > 0)).astype(np.float64)
                self.grad += self_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]
        return out
    
    def sigmoid(self):
        x = self.data
        x_safe = x * (abs(x) <= 500) + 500 * np.sign(x) * (abs(x) > 500)
        out_data = 1 / (1 + np.exp(-x_safe))
        out = Tensor(out_data, requires_grad = self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = grad * out.data * (1 - out.data)
                self.grad += self_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out
    
    def tanh(self):
        out = Tensor(np.tanh(self.data), requires_grad = self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = grad * (1 - out.data * out.data)
                self.grad += grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out
    
    def softmax(self, axis = -1):
        x = self.data
        x_max = np.max(x, axis = axis, keepdims = True)
        exp_x = np.exp(x - x_max)
        softmax_x = exp_x / np.sum(exp_x, axis = axis, keepdims = True)
        out = Tensor(softmax_x, requires_grad = self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                s = out.data
                self_grad = s * (grad - (grad * s).sum(axis = axis, keepdims = True))
                self.grad += self_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out

    def reshape(self, *shape):
        """Reshapes tensor to new dimensions."""
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                # Reshape gradient back to original shape
                self.grad += grad.reshape(self.data.shape)

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out

    def transpose(self, *axes):
        """Transposes tensor dimensions."""
        # If no axes specified, reverse all axes
        if not axes:
            axes = tuple(range(len(self.data.shape)))[::-1]
        
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                # Need to invert the permutation for gradient
                inverse_axes = [0] * len(axes)
                for i, axis in enumerate(axes):
                    inverse_axes[axis] = i
                self.grad += grad.transpose(*inverse_axes)

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out

    def squeeze(self, axis=None):
        """Remove singleton dimensions."""
        out = Tensor(self.data.squeeze(axis), requires_grad=self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                # Restore singleton dimensions for gradient
                self.grad += grad.reshape(self.data.shape)

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out

    def unsqueeze(self, axis):
        """Add singleton dimension at specified position."""
        out = Tensor(np.expand_dims(self.data, axis), requires_grad=self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                # Remove added dimension from gradient
                self.grad += grad.squeeze(axis)

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out

    def exp(self):
        """Exponential function"""
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = grad * out.data
                self.grad += self_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out

    def log(self):
        """Natural logarithm"""
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self_grad = grad / self.data
                self.grad += self_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out

    def __getitem__(self, idx):
        """Support indexing for tensors"""
        out = Tensor(self.data[idx], requires_grad=self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                full_grad = np.zeros_like(self.data)
                full_grad[idx] = grad
                self.grad += full_grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out

    def __neg__(self):
        """Support unary minus operation"""
        out = Tensor(-self.data, requires_grad=self.requires_grad)

        def grad_fn(grad):
            if self.requires_grad:
                self.grad += -grad

        if out.requires_grad:
            out.grad_fn = grad_fn
            out.children = [self]

        return out
    
    def argmax(self, axis  = None, keepdims = False):
        """Returns the indices of the maximum values alog the axis"""
        
        out_data = self.data.argmax(axis = axis)
        
        if keepdims and axis is not None:
            out_data = np.expand_dims(out_data, axis)
        
        return Tensor(out_data, requries_grad = False)
