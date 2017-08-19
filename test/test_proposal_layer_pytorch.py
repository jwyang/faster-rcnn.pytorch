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
import _init_paths
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
import torch
import cPickle as pickle
import pdb

# width, height = 37, 37
# feat_stride_np = [16]
# feat_stride_pt = 16
# anchor_scales = [8, 16, 32]

# im_info_np = np.array([[600, 600, 1.6]])
# im_info_pt = torch.FloatTensor([600, 600, 1.6]).view(1,-1)

# rpn_cls_prob_pt = torch.rand(1, 18, width, height)
# rpn_cls_prob_np = rpn_cls_prob_pt.numpy()

# rpn_cls_pred_pt = torch.rand(1, 36, width, height) * 2 - 1 
# rpn_cls_pred_np = rpn_cls_pred_pt.numpy()


# load from pickle
data = pickle.load(open('data.pkl', 'rb'))
width = data['width']
height = data['height']
feat_stride_pt = data['feat_stride_pt']
anchor_scales = data['anchor_scales']
im_info_pt = data['im_info']
rpn_cls_prob_pt = data['rpn_cls_prob_pt']
rpn_cls_pred_pt = data['rpn_cls_pred_pt']

cfg_key = 'TRAIN'

# test proposal_layer (numpy)
# def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales):

# 	x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales)
# 	return x



# rois_np = proposal_layer(rpn_cls_prob_np, rpn_cls_pred_np, im_info_np,
#                            cfg_key, feat_stride_np, anchor_scales)


# rois_np = torch.from_numpy(rois_np)
# test proposal_layer(pytorch)

from model.rpn.proposal_layer import _ProposalLayer

shift_x = np.arange(0, width) * feat_stride_pt
shift_y = np.arange(0, height) * feat_stride_pt
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                          shift_x.ravel(), shift_y.ravel())).transpose())
shifts = shifts.contiguous().float()

RPN_proposal = _ProposalLayer(feat_stride_pt, anchor_scales).cuda()


rois_pt = RPN_proposal((rpn_cls_prob_pt.cuda(), rpn_cls_pred_pt.cuda(),
                                 im_info_pt.cuda(), shifts.cuda(), cfg_key))





