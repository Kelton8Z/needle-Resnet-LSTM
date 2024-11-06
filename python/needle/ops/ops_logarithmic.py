from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=-1, keepdims=True) 
        exp_Z = array_api.exp(Z - max_z)
        return Z - max_z - array_api.log(array_api.sum(exp_Z, axis=-1, keepdims=True)) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].realize_cached_data()
        max_z = array_api.max(Z, axis=-1, keepdims=True)
        exp_Z = array_api.exp(Z - max_z)
        softmax_Z = exp_Z / array_api.sum(exp_Z, axis=-1, keepdims=True)
        
        return out_grad - array_api.sum(out_grad.numpy(), axis=-1, keepdims=True) * softmax_Z
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_original = array_api.max(Z, self.axes, keepdims=True) 
        max_z_reduce = array_api.max(Z, self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_original), self.axes)) + max_z_reduce 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        
        expand_shape = list(z.shape) 
        if self.axes:
            for axis in self.axes:
                expand_shape[axis] = 1
        else:
            for axis in range(len(expand_shape)):
                expand_shape[axis] = 1
                
        grad = exp_z*grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

