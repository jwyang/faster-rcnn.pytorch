import numpy as np
import _init_paths
import torch
import cPickle as pickle
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from model.rpn.anchor_target_layer import _AnchorTargetLayer

import pdb

data = pickle.load(open('data.pkl', 'rb'))
width = data['width']
height = data['height']
feat_stride_pt = data['feat_stride_pt']
anchor_scales = data['anchor_scales']
im_info_pt = data['im_info']
rpn_cls_score_pt = data['rpn_cls_prob_pt']
rpn_bbox_pred_pt = data['rpn_cls_pred_pt']

rpn_cls_score_np = rpn_cls_score_pt.numpy()
rpn_bbox_pred_np = rpn_bbox_pred_pt.numpy()
feat_stride_np = [16]
im_info_np = np.array([[600, 600, 1.6]])

cfg_key = 'TRAIN'

gt_boxes_np = np.array([[100, 100, 400, 300, 1]])
gt_boxes_pt = torch.from_numpy(gt_boxes_np)
gt_boxes_pt = gt_boxes_pt.view(1, 1, 5).float()

gt_ishard = np.array([0])


def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard,
	dontcare_areas, im_info, feat_stride, anchor_scales):
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        anchor_target_layer_py(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, \
        	                   im_info, feat_stride, anchor_scales)
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


########################
#  test numpy version  #
########################
dontcare_areas = None
rpn_data_np = anchor_target_layer(rpn_cls_score_np, gt_boxes_np, gt_ishard, dontcare_areas,
 	                            im_info_np, feat_stride_np, anchor_scales)


########################
# test pytorch version #
########################
num_boxes = torch.IntTensor(1, 1).fill_(1)

shift_x = np.arange(0, width) * feat_stride_pt
shift_y = np.arange(0, height) * feat_stride_pt
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                          shift_x.ravel(), shift_y.ravel())).transpose())
shifts = shifts.contiguous().float()


RPN_anchor_target = _AnchorTargetLayer(feat_stride_pt, anchor_scales)
rpn_data_pt = RPN_anchor_target((rpn_cls_score_pt, gt_boxes_pt, im_info_pt, num_boxes, shifts))

pdb.set_trace()
