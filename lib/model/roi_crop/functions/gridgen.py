# functions/add.py
import torch
from torch.autograd import Function
import numpy as np


class AffineGridGenFunction(Function):
    def __init__(self, height, width,lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/(self.height)), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/(self.width)), 0), repeats = self.height, axis = 0), 0)
        # self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/(self.height - 1)), 0), repeats = self.width, axis = 0).T, 0)
        # self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/(self.width - 1)), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        #print(self.grid)

    def forward(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            output = output.cuda()

        for i in range(input1.size(0)):
                output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output

    def backward(self, grad_output):

        grad_input1 = torch.zeros(self.input1.size())

        if grad_output.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            grad_input1 = grad_input1.cuda()
            #print('gradout:',grad_output.size())
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height*self.width, 2), 1,2), self.batchgrid.view(-1, self.height*self.width, 3))

        #print(grad_input1)
        return grad_input1

class CylinderGridGenFunction(Function):
    def __init__(self, height, width,lr=1):
        super(CylinderGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        #print self.grid

    def forward(self, input1):
        self.input1 = (1+torch.cos(input1))/2

        output = torch.zeros(torch.Size([input1.size(0), self.height, self.width, 2]) )

        if not self.input1.is_cuda:
            for i in range(self.input1.size(0)):


                x = self.input1[i][0]
                low = int(np.ceil(self.width*self.input1[i][0]))
                frac =  self.width*self.input1[i][0] - low
                interp =  frac * 2 * (1-x) + (1-frac) * 2 * (-x)

                output[i,:,:,1] = torch.zeros(self.grid[:,:,1].size())
                if low <= self.width and low > 0:
                    output[i,:,:low,1].fill_(2*(1-x))

                if low < self.width and low >= 0:
                    output[i,:,low:,1].fill_(2*(-x))

                output[i,:,:,1] = output[i,:,:,1] + self.grid[:,:,1]
                output[i,:,:,0] = self.grid[:,:,0]
        else:
            print('not implemented')
        return output

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        if not grad_output.is_cuda:
            for i in range(self.input1.size(0)):
                #print(torch.sum(grad_output[i,:,:,1],1).size())
                grad_input1[i] = -torch.sum(torch.sum(grad_output[i,:,:,1],1)) * torch.sin(self.input1[i]) / 2
        else:
            print('not implemented')
        return grad_input1 * self.lr
