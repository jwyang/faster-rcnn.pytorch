import torch
import numpy as np
import _init_paths
from model.rpn.proposal_target_layer_batch import _ProposalTargetLayer
from rpn.proposal_target_layer import proposal_target_layer

import cPickle as pickle

import pdb

data = pickle.load(open('proposal_target_layer.pkl', 'rb'))

rois_pt = data['rois']
gt_boxes_np = data['gt_boxes_np']

rois_np = rois_pt.numpy()
gt_boxes_pt = torch.from_numpy(gt_boxes_np)

n_classes = 21


########################
#  test numpy version  #
########################
gt_ishard = np.array([0, 0])
dontcare_areas = np.array([])
roi_data_np = proposal_target_layer(rois_np, gt_boxes_np)

########################
# test pytorch version #
########################
num_boxes = torch.IntTensor(1, 1).fill_(2)

RPN_proposal_target = _ProposalTargetLayer(n_classes)
roi_data_pt = RPN_proposal_target(rois_pt, gt_boxes_pt.view(1, 2, 5).float(), num_boxes)

pdb.set_trace()