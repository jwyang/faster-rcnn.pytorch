from torch.nn.modules.module import Module
import torch
from torch.autograd import Variable
import numpy as np
from ..functions.gridgen import AffineGridGenFunction

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)


class _AffineGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(_AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr
    def forward(self, input):
        # if not self.aux_loss:
        return self.f(input)
        # else:
        #     identity = torch.from_numpy(np.array([[1,0,0], [0,1,0]], dtype=np.float32))
        #     batch_identity = torch.zeros([input.size(0), 2,3])
        #     for i in range(input.size(0)):
        #         batch_identity[i] = identity
        #     batch_identity = Variable(batch_identity)
        #     loss = torch.mul(input - batch_identity, input - batch_identity)
        #     loss = torch.sum(loss,1)
        #     loss = torch.sum(loss,2)

        #       return self.f(input), loss.view(-1,1)

class CylinderGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(CylinderGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = CylinderGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr
    def forward(self, input):

        if not self.aux_loss:
            return self.f(input)
        else:
            return self.f(input), torch.mul(input, input).view(-1,1)


class AffineGridGenV2(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(AffineGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr

        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))


    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())

        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)

        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()

        output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output


class CylinderGridGenV2(Module):
    def __init__(self, height, width, lr = 1):
        super(CylinderGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.lr = lr
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
    def forward(self, input):
        self.batchgrid = torch.zeros(torch.Size([input.size(0)]) + self.grid.size() )
        #print(self.batchgrid.size())
        for i in range(input.size(0)):
            self.batchgrid[i,:,:,:] = self.grid
        self.batchgrid = Variable(self.batchgrid)

        #print(self.batchgrid.size())

        input_u = input.view(-1,1,1,1).repeat(1,self.height, self.width,1)
        #print(input_u.requires_grad, self.batchgrid)

        output0 = self.batchgrid[:,:,:,0:1]
        output1 = torch.atan(torch.tan(np.pi/2.0*(self.batchgrid[:,:,:,1:2] + self.batchgrid[:,:,:,2:] * input_u[:,:,:,:])))  /(np.pi/2)
        #print(output0.size(), output1.size())

        output = torch.cat([output0, output1], 3)
        return output


class DenseAffineGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(DenseAffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr

        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))


    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())

        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        self.batchgrid = Variable(self.batchgrid)
        #print self.batchgrid,  input1[:,:,:,0:3]
        #print self.batchgrid,  input1[:,:,:,4:6]
        x = torch.mul(self.batchgrid, input1[:,:,:,0:3])
        y = torch.mul(self.batchgrid, input1[:,:,:,3:6])

        output = torch.cat([torch.sum(x,3),torch.sum(y,3)], 3)
        return output




class DenseAffine3DGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(DenseAffine3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr

        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

        self.theta = self.grid[:,:,0] * np.pi/2 + np.pi/2
        self.phi = self.grid[:,:,1] * np.pi

        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)

        self.grid3d = torch.from_numpy(np.zeros( [self.height, self.width, 4], dtype=np.float32))

        self.grid3d[:,:,0] = self.x
        self.grid3d[:,:,1] = self.y
        self.grid3d[:,:,2] = self.z
        self.grid3d[:,:,3] = self.grid[:,:,2]


    def forward(self, input1):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.grid3d.size())

        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d

        self.batchgrid3d = Variable(self.batchgrid3d)
        #print(self.batchgrid3d)

        x = torch.sum(torch.mul(self.batchgrid3d, input1[:,:,:,0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:,:,:,4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:,:,:,8:]), 3)
        #print(x)
        r = torch.sqrt(x**2 + y**2 + z**2) + 1e-5

        #print(r)
        theta = torch.acos(z/r)/(np.pi/2)  - 1
        #phi = torch.atan(y/x)
        phi = torch.atan(y/(x + 1e-5))  + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi/np.pi


        output = torch.cat([theta,phi], 3)

        return output





class DenseAffine3DGridGen_rotate(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(DenseAffine3DGridGen_rotate, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr

        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

        self.theta = self.grid[:,:,0] * np.pi/2 + np.pi/2
        self.phi = self.grid[:,:,1] * np.pi

        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)

        self.grid3d = torch.from_numpy(np.zeros( [self.height, self.width, 4], dtype=np.float32))

        self.grid3d[:,:,0] = self.x
        self.grid3d[:,:,1] = self.y
        self.grid3d[:,:,2] = self.z
        self.grid3d[:,:,3] = self.grid[:,:,2]


    def forward(self, input1, input2):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.grid3d.size())

        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d

        self.batchgrid3d = Variable(self.batchgrid3d)

        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())

        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        self.batchgrid = Variable(self.batchgrid)

        #print(self.batchgrid3d)

        x = torch.sum(torch.mul(self.batchgrid3d, input1[:,:,:,0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:,:,:,4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:,:,:,8:]), 3)
        #print(x)
        r = torch.sqrt(x**2 + y**2 + z**2) + 1e-5

        #print(r)
        theta = torch.acos(z/r)/(np.pi/2)  - 1
        #phi = torch.atan(y/x)
        phi = torch.atan(y/(x + 1e-5))  + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi/np.pi

        input_u = input2.view(-1,1,1,1).repeat(1,self.height, self.width,1)

        output = torch.cat([theta,phi], 3)

        output1 = torch.atan(torch.tan(np.pi/2.0*(output[:,:,:,1:2] + self.batchgrid[:,:,:,2:] * input_u[:,:,:,:])))  /(np.pi/2)
        output2 = torch.cat([output[:,:,:,0:1], output1], 3)

        return output2


class Depth3DGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(Depth3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr

        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

        self.theta = self.grid[:,:,0] * np.pi/2 + np.pi/2
        self.phi = self.grid[:,:,1] * np.pi

        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)

        self.grid3d = torch.from_numpy(np.zeros( [self.height, self.width, 4], dtype=np.float32))

        self.grid3d[:,:,0] = self.x
        self.grid3d[:,:,1] = self.y
        self.grid3d[:,:,2] = self.z
        self.grid3d[:,:,3] = self.grid[:,:,2]


    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.grid3d.size())

        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d

        self.batchgrid3d = Variable(self.batchgrid3d)

        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.grid.size())

        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid

        self.batchgrid = Variable(self.batchgrid)

        x = self.batchgrid3d[:,:,:,0:1] * depth + trans0.view(-1,1,1,1).repeat(1, self.height, self.width, 1)

        y = self.batchgrid3d[:,:,:,1:2] * depth + trans1.view(-1,1,1,1).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:,:,:,2:3] * depth
        #print(x.size(), y.size(), z.size())
        r = torch.sqrt(x**2 + y**2 + z**2) + 1e-5

        #print(r)
        theta = torch.acos(z/r)/(np.pi/2)  - 1
        #phi = torch.atan(y/x)
        phi = torch.atan(y/(x + 1e-5))  + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi/np.pi

        #print(theta.size(), phi.size())


        input_u = rotate.view(-1,1,1,1).repeat(1,self.height, self.width,1)

        output = torch.cat([theta,phi], 3)
        #print(output.size())

        output1 = torch.atan(torch.tan(np.pi/2.0*(output[:,:,:,1:2] + self.batchgrid[:,:,:,2:] * input_u[:,:,:,:])))  /(np.pi/2)
        output2 = torch.cat([output[:,:,:,0:1], output1], 3)

        return output2





class Depth3DGridGen_with_mask(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False, ray_tracing = False):
        super(Depth3DGridGen_with_mask, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.ray_tracing = ray_tracing

        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

        self.theta = self.grid[:,:,0] * np.pi/2 + np.pi/2
        self.phi = self.grid[:,:,1] * np.pi

        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)

        self.grid3d = torch.from_numpy(np.zeros( [self.height, self.width, 4], dtype=np.float32))

        self.grid3d[:,:,0] = self.x
        self.grid3d[:,:,1] = self.y
        self.grid3d[:,:,2] = self.z
        self.grid3d[:,:,3] = self.grid[:,:,2]


    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.grid3d.size())

        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d

        self.batchgrid3d = Variable(self.batchgrid3d)

        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.grid.size())

        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid

        self.batchgrid = Variable(self.batchgrid)

        if depth.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            self.batchgrid3d = self.batchgrid3d.cuda()


        x_ = self.batchgrid3d[:,:,:,0:1] * depth + trans0.view(-1,1,1,1).repeat(1, self.height, self.width, 1)

        y_ = self.batchgrid3d[:,:,:,1:2] * depth + trans1.view(-1,1,1,1).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:,:,:,2:3] * depth
        #print(x.size(), y.size(), z.size())

        rotate_z = rotate.view(-1,1,1,1).repeat(1,self.height, self.width,1) * np.pi

        x = x_ * torch.cos(rotate_z) - y_ * torch.sin(rotate_z)
        y = x_ * torch.sin(rotate_z) + y_ * torch.cos(rotate_z)


        r = torch.sqrt(x**2 + y**2 + z**2) + 1e-5

        #print(r)
        theta = torch.acos(z/r)/(np.pi/2)  - 1
        #phi = torch.atan(y/x)

        if depth.is_cuda:
            phi = torch.atan(y/(x + 1e-5))  + np.pi * x.lt(0).type(torch.cuda.FloatTensor) * (y.ge(0).type(torch.cuda.FloatTensor) - y.lt(0).type(torch.cuda.FloatTensor))
        else:
            phi = torch.atan(y/(x + 1e-5))  + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))


        phi = phi/np.pi

        output = torch.cat([theta,phi], 3)
        return output
