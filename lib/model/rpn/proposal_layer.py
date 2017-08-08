# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import yaml
from model.utils.config import cfg
from generate_anchors import generate_anchors
from bbox_transform import bbox_transform_inv, clip_boxes
from model.nms.nms_wrapper import nms

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales))).float()
        self._num_anchors = self._anchors.size(0)

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        rpn_cls_prob = input[0]
        rpn_bbox_pred = input[1]
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = input[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = input[1].data
        im_info = input[2].data[0, :]

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.size(2), scores.size(3)

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().float()
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.size(0)

        anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = anchors.view(K * A, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        # bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(-1, 4)

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        # scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(-1, 1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2]).squeeze()
        keep_idx = torch.nonzero(keep).squeeze()
        proposals = proposals[keep_idx, :]
        scores = scores[keep_idx, :]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        _, order = torch.sort(scores, 0, True)
        order = order.squeeze()
        # order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order, :]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)

        # TODO:jwyang:Compile nms for pytorch, and replace the original one
        # TODO:jwyang:NMS is slow, make it faster !!!!!
        proposals_np = proposals.numpy()
        scores_np = scores.numpy()
        keep_np = nms(np.hstack((proposals_np, scores_np)), nms_thresh)
        keep = torch.from_numpy(np.asarray(keep_np))

        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep, :]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        # blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        # top[0].reshape(*(blob.shape))
        # top[0].data[...] = blob
        batch_inds = torch.FloatTensor(proposals.size(0), 1).zero_()
        output = torch.cat((batch_inds, proposals), 1)

        # [Optional] output scores blob
        # if len(top) > 1:
        #     top[1].reshape(*(scores.shape))
        #     top[1].data[...] = scores

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = ((ws >= min_size) & (hs >= min_size))
    return keep
