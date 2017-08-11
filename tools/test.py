import torch
import torch.nn as nn
from DataParallelModified import DataParallelModified
from torch.autograd import Variable
import pdb

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.block1 = nn.Linear(10, 20)
        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block3 = nn.Linear(20, 20)

        self.tmp_tensor = torch.Tensor(1,10)

    def forward(self, x):

        pdb.set_trace()

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

model = nn.DataParallel(Model().cuda())

data1 = Variable(torch.FloatTensor(10,10).cuda())
data2 = Variable(torch.FloatTensor(10,6).cuda())

out = model.forward(data1)