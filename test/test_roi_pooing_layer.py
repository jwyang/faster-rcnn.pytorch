import torch
import numpy as np
import pdb
import torch
from torch.autograd import Variable
import torch.nn as nn
import _init_paths
from roi_pooling.modules.roi_pool import RoIPool
from model.roi_pooling.modules.roi_pool import _RoIPooling
import torch.nn.functional as F
from torch.autograd import gradcheck
gt_boxes_np = np.array([[0, 419.20001221,  336.        ,  516.79998779,  540.79998779],
                       [0, 262.3999939 ,  420.79998779,  403.20001221,  593.59997559],
                       [0,    6.4000001 ,  388.79998779,  105.59999847,  596.79998779],
                       [0,  384.        ,  308.79998779,  470.3999939 ,  476.79998779],
                       [0,  441.6000061 ,  296.        ,  497.6000061 ,  350.3999939]], 
                       dtype=float)


gt_boxes_pt = torch.from_numpy(gt_boxes_np).float()
features = torch.rand(1, 256, 37, 37).float()

gt_boxes_pt1 = Variable(gt_boxes_pt.cuda())
features1 = Variable(features.cuda(), requires_grad = True)

gt_boxes_pt2 = Variable(gt_boxes_pt.cuda())
features2 = Variable(features.cuda(), requires_grad = True)

pdb.set_trace()
########################################
# test single image roi pooloing layer #
########################################

roi_pool = RoIPool(7, 7, 1.0/16).cuda()
pooled_features = roi_pool(features1, gt_boxes_pt1)
s1 = torch.sum(pooled_features)
loss = F.smooth_l1_loss(s1, Variable(torch.Tensor(1).zero_().cuda()))
loss.backward()

#######################################
# test multi image roi pooloing layer #
#######################################

RCNN_roi_pool = _RoIPooling(7, 7, 1.0/16.0).cuda()
pooled_feat = RCNN_roi_pool(features2, gt_boxes_pt2)
s2 = torch.sum(pooled_feat)
loss = F.smooth_l1_loss(s2, Variable(torch.Tensor(1).zero_().cuda()))
loss.backward()

# test = gradcheck(RCNN_roi_pool, (features, gt_boxes_pt), eps=1e-6, atol=1e-4)
# print(test)

