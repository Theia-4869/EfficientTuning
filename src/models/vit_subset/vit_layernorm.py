import torch
from torch.nn import Module, LayerNorm, init

from torch import Tensor


class FrontModule(Module):
    __constants__ = ['module']
    module: Module

    def __init__(self, module: Module) -> None:
        super(FrontModule, self).__init__()
        self.normalized_shape = module.normalized_shape
        self.eps = module.eps
        self.elementwise_affine = module.elementwise_affine
        
        self.pre = module
        self.new = LayerNorm(self.normalized_shape, self.eps, self.elementwise_affine)
        
        for _, param in self.pre.named_parameters():
            param.requires_grad = True
        for _, param in self.new.named_parameters():
            param.requires_grad = True

    def forward(self, input: Tensor) -> Tensor:
        return self.pre(self.new(input))

class BackModule(Module):
    __constants__ = ['module']
    module: Module

    def __init__(self, module: Module) -> None:
        super(BackModule, self).__init__()
        self.normalized_shape = module.normalized_shape
        self.eps = module.eps
        self.elementwise_affine = module.elementwise_affine
        
        self.pre = module
        self.new = LayerNorm(self.normalized_shape, self.eps, self.elementwise_affine)
        
        for _, param in self.pre.named_parameters():
            param.requires_grad = True
        for _, param in self.new.named_parameters():
            param.requires_grad = True

    def forward(self, input: Tensor) -> Tensor:
        return self.new(self.pre(input))

class SideModule(Module):
    __constants__ = ['module']
    module: Module

    def __init__(self, module: Module) -> None:
        super(SideModule, self).__init__()
        self.normalized_shape = module.normalized_shape
        self.eps = module.eps
        self.elementwise_affine = module.elementwise_affine
        
        self.pre = module
        self.new = LayerNorm(self.normalized_shape, self.eps, self.elementwise_affine)
        init.zeros_(self.new.weight)
        init.zeros_(self.new.bias)
        
        for _, param in self.pre.named_parameters():
            param.requires_grad = True
        for _, param in self.new.named_parameters():
            param.requires_grad = True

    def forward(self, input: Tensor) -> Tensor:
        return (self.pre(input) + self.new(input))
