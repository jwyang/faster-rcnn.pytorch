from torch.nn.modules.module import Module
from ...roi_pooling.functions.roi_pool import RoIPoolFunction
from ..functions.roioffset_pool import RoIOffsetPoolFunction
import torch.nn as nn
import torch

class _RoIOffsetPooling(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, use_offset=True, offset=None):
        super(_RoIOffsetPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        fc_channel = self.pooled_width * self.pooled_height
        self.fc = nn.Linear(fc_channel, 2*fc_channel)
        self.gamma = 0.1
        self.use_offset = use_offset
        self.offset = offset
           
    def forward(self, features, rois):
        n,c,h,w = features.size()
        pooled = RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)
        pooled = pooled.view(rois.size(0)*c,-1)
        offset = self.fc(pooled) * self.gamma
        offset = offset.view(rois.size(0), c, self.pooled_height, self.pooled_width, 2)
        if not self.use_offset:
            offset *= 0
        
        if self.offset is not None:
            offset += self.offset

        return RoIOffsetPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois, offset)