from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time, os


def tile(arr, copy, axis):
    return np.concatenate([arr] * copy, axis=axis)

class Module(object):
    def __init__(self, trainable=False):
        self.trainable = trainable
        pass
    
    def forward(self, x):
        raise NotImplementedError
        
    def backward(self, grad, optimizer=None):
        raise NotImplementedError
        
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)


class Sequence(Module):
    def __init__(self, modules):
        self._modules = modules
        
    def forward(self, inpt):
        for module in self._modules:
            inpt = module(inpt)
            if module.trainable:
                self.trainable = True
        return inpt
    
    def backward(self, grad, optimizer=None):
        for module in self._modules[::-1]:
            if module.trainable:
                grad = module.backward(grad, optimizer)
            else:
                grad = module.backward(grad)
        return grad
    
    def modules(self):
        return self._modules
    
    def trainable_modules(self):
        return [i for i in self._modules if i.trainable]


class Linear(Module):
    def __init__(self, in_channel, out_channel, eps=1e-4):
        super(Linear, self).__init__(trainable=True)
        self.w = np.random.randn(in_channel, out_channel)
        self.w = self.w/np.max(self.w)*eps
        self.b = np.zeros((1, out_channel))
        self.x = None
        
    def _set_params(self, params):
        w, b = params
        self.w = w
        if b is not None:
            self.b = b
        
    def forward(self, x):
        out = x.dot(self.w.T) + self.b
        self.x = x
        return out
    
    def backward(self, grad, optimizer=None):
        dw = self.x.T.dot(grad)
        db = np.sum(grad, axis=0)
        # update parameters
        if optimizer is not None:
            self.w = optimizer(self.w, dw)
            self.b = optimizer(self.b, db)
        
        dx = grad.dot(self.w)
        dx = np.reshape(dx, self.x.shape)
        return dx


class ReLU(Module):
    def __init__(self, alpha=0):
        super(ReLU, self).__init__()
        self.alpha = alpha
        self.x = None
        
    def forward(self, x):
        out = x.copy()
        if self.alpha > 0:
            out[out<0] = self.alpha*x
        else:
            out[out<0] = 0
        self.x = x
        return out
    
    def backward(self, grad):
        dx = grad.copy()
        dx[self.x < 0] = 0
        return dx


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.s = None
        
    def forward(self, x):
        self.s = 1/(1 + np.exp(-x))
        return self.s
    
    def backward(self, grad):
        return grad * (self.s * (1-self.s))

    
class Softmax(Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.s = None
        self.dim = dim
        self.squeeze_len = None
        self.y = None
        
    def forward(self, x):
        if self.dim < 0:
            self.dim = len(x.shape)+self.dim
        self.squeeze_len = x.shape[self.dim]
        y = np.exp(x)
        sm = np.sum(y, axis=self.dim, keepdims=True)
        s = y/sm
        
        self.s = s
        self.y = y
        return s
    
    def backward(self, grad):
        sm = np.sum(self.y, axis=self.dim, keepdims=True)
        temp1 = grad * (1/sm) * self.y
        temp2 = grad * self.y * (-sm**(-2)) * self.y
        return temp1 + temp2


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=1, eps=1e-4, bias=True):
        super(Conv2d, self).__init__(trainable=True)
        self.ic = in_channels
        self.oc = out_channels
        self.k = kernel_size
        self.s = stride
        self.p = pad
        self.W = np.random.rand(self.oc,self.ic,self.k,self.k)
        self.W = self.W * eps
        self.b = np.zeros((self.oc,1))
        self.bias = bias
        self.offset = None
        
    def _set_params(self, params):
        W, b = params
        self.W = W
        if b is not None:
            self.b = b
        
    def forward(self, X):

        NF, CF, HF, WF = self.W.shape
        NX, DX, HX, WX = X.shape
        h_out = int((HX - HF + 2 * self.p) / self.s + 1)
        w_out = int((WX - WF + 2 * self.p) / self.s + 1)

        X_col = self.im2col_indices(X)
        self.X_col = X_col
        W_col = self.W.reshape(NF, -1)

        out = W_col.dot( self.X_col ) + self.b
        out = out.reshape(NF, h_out, w_out, NX)
        out = out.transpose(3, 0, 1, 2)

        return out

    def backward(self, dout, optimizer=None):

        NF, CF, HF, WF = self.W.shape
        
        if self.bias:
            db = np.sum(dout, axis=(0, 2, 3))
            db = db.reshape(NF, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(NF, -1)
        dW = dout_reshaped.dot( self.X_col.T )
        dW = dW.reshape(self.W.shape)
        
        if optimizer is not None:
            if self.bias:
                self.b = optimizer(self.b, db)
            self.W = optimizer(self.W, dW)

        W_reshape = self.W.reshape(NF, -1)
        dX_col = W_reshape.T.dot( dout_reshaped )
        dX = self.col2im_indices(dX_col)

        return dX

    def get_im2col_indices(self):

        padding, stride, field_height, field_width, x_shape = self.p, self.s, self.k, self.k, self.x_shape
        N, C, H, W = x_shape
        out_height = int((H + 2 * padding - field_height) / stride + 1)
        out_width = int((W + 2 * padding - field_width) / stride + 1)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
        return (k.astype(int), i.astype(int), j.astype(int))


    def im2col_indices(self, x):
        p, stride, field_height, field_width = self.p, self.s, self.k, self.k
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_im2col_indices()
        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    def col2im_indices(self, cols):
        field_height, field_width, padding, stride = self.k, self.k, self.p, self.s
        N, C, H, W = self.x_shape
        print(self.x_shape)
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.get_im2col_indices()
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]

class DeformableConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=1, eps=1e-4, bias=True):
        super(DeformableConv2d, self).__init__(trainable=True)
        self.ic = in_channels
        self.oc = out_channels
        self.k = kernel_size
        self.s = stride
        self.p = pad
        self.W = np.random.rand(self.oc,self.ic,self.k,self.k)
        self.b = np.zeros((self.oc,1))
        self.bias = bias
        self.X = None
        
    def _set_params(self, params):
        W, b = params
        self.W = W
        if b is not None:
            self.b = b
        
    def forward(self, X, offset):
        NF, CF, HF, WF = self.W.shape
        NX, DX, HX, WX = X.shape
        self.X = X
        h_out = int((HX - HF + 2 * self.p) / self.s + 1)
        w_out = int((WX - WF + 2 * self.p) / self.s + 1)
        
        self.offset = offset
        X_col = self.im2col_indices(X, offset)
        W_col = self.W.reshape(NF, -1)

        out = W_col.dot( X_col ) + self.b
        out = out.reshape(NF, h_out, w_out, NX)
        out = out.transpose(3, 0, 1, 2)

        return out

    def backward(self, dout, optimizer=None):

        NF, CF, HF, WF = self.W.shape
        
        X_col = self.im2col_indices(self.X, self.offset)
        
        if self.bias:
            db = np.sum(dout, axis=(0, 2, 3))
            db = db.reshape(NF, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(NF, -1)
        dW = dout_reshaped.dot( X_col.T )
        dW = dW.reshape(self.W.shape)
        
        if optimizer is not None:
            if self.bias:
                self.b = optimizer(self.b, db)
            self.W = optimizer(self.W, dW)

        W_reshape = self.W.reshape(NF, -1)
        dX_col = W_reshape.T.dot( dout_reshaped )
        dX = self.col2im_indices(dX_col)

        return dX

    def get_im2col_indices(self, offset):
        padding, stride, field_height, field_width, x_shape = self.p, self.s, self.k, self.k, self.X.shape
        N, C, H, W = x_shape
        offset = np.resize(offset, (N,self.k**2,2,H*W))
        offset = tile(offset, C, 1)
        out_height = int((H + 2 * padding - field_height) / stride + 1)
        out_width = int((W + 2 * padding - field_width) / stride + 1)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
        
        i = tile(np.expand_dims(i, axis=0), N, 0) + offset[:,:,0,:]
        j = tile(np.expand_dims(j, axis=0), N, 0) + offset[:,:,1,:]
        k = tile(np.expand_dims(k, axis=0), N, 0)
        n = np.arange(N)[:,None,None]
        return (n, k, i, j)

    def index_bilinear(self, x, indices):
        N, C, H, W = x.shape
        n, k, i, j = indices
        i[i>H-1] = H-1
        i[i<0] = 0
        j[j>W-1] = W-1
        j[j<0] = 0
        i_floor, j_floor = np.floor(i).astype(int), np.floor(j).astype(int)
        i_ceil, j_ceil = i_floor+1, j_floor+1
        i_ceil[i_ceil>H-1] = H-1
        j_ceil[j_ceil>W-1] = W-1
        
        out = np.zeros(i.shape)
        print( (1-np.abs(i_floor-i)).shape, (1-np.abs(j_ceil-j)).shape, (x[n,k,i_floor, j_ceil]).shape )
        out += ((1-np.abs(i_ceil-i)) * (1-np.abs(j_ceil-j))) * (x[n,k,i_ceil, j_ceil])
        out += ((1-np.abs(i_floor-i)) * (1-np.abs(j_ceil-j))) * (x[n,k,i_floor, j_ceil])
        out += ((1-np.abs(i_ceil-i)) * (1-np.abs(j_floor-j))) * (x[n,k,i_ceil, j_floor])
        out += ((1-np.abs(i_floor-i)) * (1-np.abs(j_floor-j))) * (x[n,k,i_floor, j_floor])
        return out
        
    def im2col_indices(self, x, offset):
        
        p, stride, field_height, field_width = self.p, self.s, self.k, self.k
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        indices = self.get_im2col_indices(offset)
        cols = self.index_bilinear(x_padded, indices)
        print('cols', cols.shape)
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    def col2im_indices(self, cols):
        field_height, field_width, padding, stride = self.k, self.k, self.p, self.s
        N, C, H, W = self.X.shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        n, k, i, j = self.get_im2col_indices(self.offset)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, n, k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]
        

from modules import ConvOffset2d

print('Testing...')
torch.manual_seed(1)


N, inC, inH, inW = 12, 6, 100, 100
outC, outH, outW = 4, 100, 100
kH, kW = 3, 3

conv = nn.Conv2d(
    inC,
    2 * kH * kW,
    kernel_size=(kH, kW),
    stride=(1, 1),
    padding=(1, 1),
    bias=False).cuda()

conv_offset2d = ConvOffset2d(
    inC,
    outC, (kH, kW),
    stride=1,
    padding=1).cuda()

# conv_offset2d = nn.Conv2d(
#     inC,
#     outC,
#     kernel_size=(kH, kW),
#     stride=(1,1),
#     padding=(1,1),
#     bias=False).cuda()

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

offset_weight = conv.weight.cpu().data.numpy()
deform_weight = conv_offset2d.weight.cpu().data.numpy()

Conv = Conv2d(inC, 2 * kH * kW, kH, bias=False)
DeformConv = DeformableConv2d(inC, outC, kH, bias=False)
Conv._set_params([offset_weight, None])
DeformConv._set_params([deform_weight, None])

inputs = Variable(torch.randn(N, inC, inH, inW), requires_grad=True).cuda()
offset = conv(inputs)
output = conv_offset2d(inputs, offset)
loss = torch.sum(output)

output.register_hook(save_grad('output'))
inputs.register_hook(save_grad('inputs'))

loss.backward()
print('pytorch grad', grads['inputs'][0,0,0], grads['output'][0,0,0])

inputs = inputs.data.cpu().numpy()
offset = Conv(inputs)
output = DeformConv(inputs, offset)
output_grad = DeformConv.backward(np.ones(output.shape))
input_grad = Conv.backward(output_grad)
print('deformable grad', input_grad[0,0,0], output_grad[0,0,0])

# print(output.size())
