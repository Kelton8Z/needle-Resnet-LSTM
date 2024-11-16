"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, BatchNorm2d, ReLU

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = (kernel_size - 1) // 2
        kernel_sq = kernel_size**2
        self.weight = Parameter(init.kaiming_uniform(in_channels*kernel_sq,
                                           out_channels*kernel_sq,
                                           shape=(kernel_size, kernel_size, in_channels, out_channels),
                                           device=device))
        if bias:
            interval = 1/(in_channels*kernel_sq)**0.5
            self.bias = Parameter(init.rand(self.out_channels, low=-interval, high=interval, device=device))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        nhwc_x = x.transpose((1, 2)).transpose((2, 3)) # NCHW -> NHWC
        out = ops.conv(nhwc_x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias:
            out += self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(out.shape)
        out = out.transpose((3, 2)).transpose((2, 1))
        return out
        ### END YOUR SOLUTION
        
class ConvBN(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride=stride, device=device)
        self.bn = BatchNorm2d(out_channels, device=device)
        self.relu = ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x