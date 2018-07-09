from torch.nn.modules.module import Module
import sys
from model.psroi_pooling.functions.psroi_pooling import PSRoIPoolingFunction


class PSRoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        super(PSRoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

    def forward(self, features, rois):
        return PSRoIPoolingFunction(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size, self.output_dim)(features, rois)

if __name__ == '__main__':
    import torch
    import numpy as np
    from torch.autograd import Variable
    from model.roi_pooling.modules.roi_pool import _RoIPooling

    input = torch.randn(2, 21*7*7, 50, 72)
    rois = torch.from_numpy(
        np.array([
            [0.0000, 350.6689, 211.0240, 779.0886, 777.7496],
            [0.0000, 744.0627, 277.4919, 988.4307, 602.7589],
            [1.0000, 350.6689, 211.0240, 779.0886, 777.7496],
            [1.0000, 744.0627, 277.4919, 988.4307, 602.7589],
        ])
    ).float()

    pool = PSRoIPool(7, 7, 1/16.0, 7, 21)
    input = Variable(input.cuda())
    rois = Variable(rois.cuda())
    print(rois.size(), input.size())
    print(input)
    out = pool(input, rois)
    print(out)
    print(out.size())

    print('============================')
    roi_pool = _RoIPooling(7, 7, 1/16.0)
    out = roi_pool(input, rois.view(-1, 5))
    print(out)
    print(out.size())