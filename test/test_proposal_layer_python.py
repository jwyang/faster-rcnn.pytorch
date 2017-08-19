import numpy as np
import _init_paths
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
import torch
import cPickle as pickle
import pdb

data = pickle.load(open('data.pkl', 'rb'))
width = data['width']
height = data['height']
feat_stride_pt = data['feat_stride_pt']
anchor_scales = data['anchor_scales']
im_info_pt = data['im_info']
rpn_cls_prob_pt = data['rpn_cls_prob_pt']
rpn_cls_pred_pt = data['rpn_cls_pred_pt']


rpn_cls_pred_np = rpn_cls_pred_pt.numpy()
rpn_cls_prob_np = rpn_cls_prob_pt.numpy()
feat_stride_np = [16]
im_info_np = np.array([[600, 600, 1.6]])

cfg_key = 'TRAIN'

# test proposal_layer (numpy)
def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales):

	x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales)
	return x


rois_np = proposal_layer(rpn_cls_prob_np, rpn_cls_pred_np, im_info_np,
                           cfg_key, feat_stride_np, anchor_scales)

rois_np = torch.from_numpy(rois_np)

pdb.set_trace()