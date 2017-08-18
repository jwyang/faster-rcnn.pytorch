# test case for the RPN layer. 
# proposal_layer.py

import torch
import numpy as np

"""
Input (python)
-------------
rpn_cls_prob: (1 x 18 x width x height) (0, 1)
rpn_cls_pred: (1 x 36 x width x height) (-1, 1)
im_info: (1, 3) [[600, 600, 1.6]] 
cfg_key:
_feat_stride:
anchor_scales:

Input (pytorch)
---------------
rpn_cls_prob: (1 x 18 x width x height) (0, 1)
rpn_cls_pred: (1 x 36 x width x height) (-1, 1)
im_info: [[600, 600, 1.6]] 
shifts:
cfg_key:

Initilize
----------
feat_stride:
scales:

"""
width, heigth = 37, 37
feat_stride_np = [16]
feat_stride_pt = 16
anchor_scales = [8, 16, 32]

im_info_np = np.array([[600, 800, 1.6]])
im_info_pt = torch.FloatTensor([600, 600, 1.6])

rpn_cls_prob_pt = torch.random(1, 18, width, height)
rpn_cls_prob_np = rpn_cls_prob_pt.numpy()

rpn_cls_pred_pt = torch.random(1, 36, width, height) * 2 - 1 
rpn_cls_pred_np = rpn_cls_pred_pt.numpy()

cfg_key = 'TRAIN'










# proposal_target_layer.py











# anchor_target_layer.py










