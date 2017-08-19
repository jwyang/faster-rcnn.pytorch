import torch
import numpy as np
import cPickle as pickle

width, height = 37, 37
feat_stride_np = [16]
feat_stride_pt = 16
anchor_scales = [8, 16, 32]

im_info_np = np.array([[600, 600, 1.6]])
im_info_pt = torch.FloatTensor([600, 600, 1.6]).view(1,-1)

rpn_cls_prob_pt = torch.rand(1, 18, width, height)
rpn_cls_prob_np = rpn_cls_prob_pt.numpy()

rpn_cls_pred_pt = torch.rand(1, 36, width, height) * 2 - 1 
rpn_cls_pred_np = rpn_cls_pred_pt.numpy()

data = {}
data['width'] = width
data['height'] = height
data['feat_stride_pt'] = feat_stride_pt
data['anchor_scales'] = anchor_scales
data['im_info'] = im_info_pt
data['rpn_cls_prob_pt'] = rpn_cls_prob_pt
data['rpn_cls_pred_pt'] = rpn_cls_pred_pt

pickle.dump(data, open('data_proposal_layer.pkl', 'w'))


rois = torch.zeros(1, 2000, 5)
rois[:,:,1:5] = torch.rand(1, 2000, 4) * 600

gt_boxes_np = np.array([[419.20001221,  336.        ,  516.79998779,  540.79998779,    9.],
				[262.3999939 ,  420.79998779,  403.20001221,  593.59997559,    9. ]])

gt_boxes_py = torch.from_numpy(gt_boxes_np)

gt_ishard = [0, 0]
dont_care_areas = []

data = {}
data['rois'] = rois
data['gt_boxes_np'] = gt_boxes_np

pickle.dump(data, open('proposal_target_layer.pkl', 'w'))

