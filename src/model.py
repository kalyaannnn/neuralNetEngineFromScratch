from typing import List, Dict, Any
from .tensor import Tensor

class Module:
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, 'Module'] = {}
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")

    def parameters(self) -> List[Tensor]:
        """Returns all parameters in the module and submodules"""
        params = []
        # Add parameters from current module
        params.extend(self._parameters.values())
        # Add parameters from submodules
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Tensor):
            # Initialize _parameters if not exists
            if '_parameters' not in self.__dict__:
                self._parameters = {}
            self._parameters[name] = value
        elif isinstance(value, Module):
            # Initialize _modules if not exists
            if '_modules' not in self.__dict__:
                self._modules = {}
            self._modules[name] = value
        object.__setattr__(self, name, value)