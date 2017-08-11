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
import numpy.random as npr

from model.utils.config import cfg
from generate_anchors import generate_anchors
from bbox_transform import bbox_transform, clip_boxes, bbox_overlaps1

import pdb

DEBUG = False

class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales))).float()
        self._num_anchors = self._anchors.size(0)

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap
        rpn_cls_score = input[0]
        gt_boxes = input[1].data
        im_info = input[2][0].data

        # TODO this should be equal to GPU number
        # assert input[0].size(1) == 1, \
        #     'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.size()', gt_boxes.size()
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        # shift_x = np.arange(0, width) * self._feat_stride
        # shift_y = np.arange(0, height) * self._feat_stride
        # shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
        #                    shift_x.ravel(), shift_y.ravel())).transpose()
        
        # -- torch version ----- 
        shift_x = gt_boxes.new(width)
        shift_x.copy_(torch.arange(0, width))
        shift_x = shift_x * self._feat_stride # Check: feat_stride only has one value.
        
        shift_y = gt_boxes.new(height)
        shift_y.copy_(torch.arange(0, height))
        shift_y = shift_y * self._feat_stride # Check: feat_stride only has one value.        
        
        shifts = torch.stack([shift_x.repeat(height), 
                            shift_y.repeat(width,1).t().contiguous().view(-1), 
                            shift_x.repeat(height), 
                            shift_y.repeat(width,1).t().contiguous().view(-1)],1)

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(shifts) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)

        total_anchors = int(K * A)

        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < im_info[1] + self._allowed_border) &
                (all_anchors[:, 3] < im_info[0] + self._allowed_border))
        inds_inside = torch.nonzero(keep).squeeze()

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', inds_inside.size()

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.size', anchors.size()

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = shifts.new(inds_inside.size(0)).fill_(-1)

        #labels = np.empty((len(inds_inside), ), dtype=np.float32)
        #labels.fill(-1)


        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        # gt_boxes_np = gt_boxes.numpy()
        # overlaps = bbox_overlaps(
        #    np.ascontiguousarray(anchors.numpy(), dtype=np.float),
        #    np.ascontiguousarray(gt_boxes_np, dtype=np.float))

        overlaps = bbox_overlaps1(anchors, gt_boxes[:,:4].contiguous())

        # argmax_overlaps = overlaps.argmax(axis=1)
        # max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        # gt_argmax_overlaps = overlaps.argmax(axis=0)
        # gt_max_overlaps = overlaps[gt_argmax_overlaps,
        #                            np.arange(overlaps.shape[1])]
        # gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        #_, argmax_overlaps1 = overlaps1.max(1)
        #max_overlaps = overlaps1[np.arange(len(inds_inside)), argmax_overlaps1]
        max_overlaps, argmax_overlaps = torch.max(overlaps, 1)
        gt_max_overlaps, _ = torch.max(overlaps, 0)
        
        keep = torch.sum(overlaps.eq(gt_max_overlaps.expand_as(overlaps)), 1)

        gt_argmax_overlaps = torch.nonzero(keep).squeeze()


        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchgitor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0


        # TODO: Check the differences between pytorch and pycaffe code here
        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        
        # fg_inds = np.where(labels == 1)[0]        
        # if len(fg_inds) > num_fg:
        #    disable_inds = npr.choice(
        #        fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        #    labels[disable_inds] = -1

        fg_inds = torch.nonzero(labels == 1).squeeze()
        if fg_inds.size(0) > num_fg:
            rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
            disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - (labels == 1).sum()
        # bg_inds = np.where(labels == 0)[0]
        # if len(bg_inds) > num_bg:
        #    disable_inds = npr.choice(
        #        bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        #    labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))
        
        bg_inds = torch.nonzero(labels == 0).squeeze()
        if bg_inds.size(0) > num_bg:
            rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()
            disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
            labels[disable_inds] = -1

        #bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        #bbox_targets = _compute_targets(anchors.numpy(), gt_boxes_np[argmax_overlaps.numpy(), :])

        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        # assign the bbox inside weights.
        #bbox_inside_weights = np.zeros((len(inds_inside.numpy()), 4), dtype=np.float32)
        #bbox_inside_weights[labels.numpy() == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        # TODO: the RPN_BBOX_INSIDE_WEIGHTS is [1, 1, 1, 1], use 1 to assign all the weight.
        # Is this fine ?          
        

        bbox_inside_weights = gt_boxes.new(inds_inside.size(0), 4).zero_()
        bbox_inside_weights[torch.nonzero(labels==1).squeeze(),:] = \
                            cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]
        

        bbox_outside_weights = gt_boxes.new(inds_inside.size(0), 4).zero_()
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = (labels > 0).sum()
            positive_weights = 1.0 / num_examples
            negative_weights = 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))            

            positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT / (labels == 1).sum()
            negative_weights = (1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / (labels == 0).sum()

        bbox_outside_weights[torch.nonzero(labels==1).squeeze()] = positive_weights
        bbox_outside_weights[torch.nonzero(labels==0).squeeze()] = negative_weights


        #bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        #if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
        #    num_examples = np.sum(labels >= 0)
        #    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        #    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        #else:
        #    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
        #            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        #    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
        #                        np.sum(labels == 1))
        #    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
        #                        np.sum(labels == 0))
        #bbox_outside_weights[labels == 1, :] = positive_weights
        #bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        outputs = []
        # labels
        # labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        # labels = labels.reshape((1, 1, A * height, width))
        # outputs.append(torch.from_numpy(labels))
        # top[0].reshape(*labels.shape)
        # top[0].data[...] = labels

        labels = labels.view(1, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(1, 1, A * height, width)
        outputs.append(labels)

        # bbox_targets
        # bbox_targets = bbox_targets \
        #    .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        # outputs.append(torch.from_numpy(bbox_targets))
        # top[1].reshape(*bbox_targets.shape)
        # top[1].data[...] = bbox_targets
        bbox_targets = bbox_targets.view(1, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)


        # bbox_inside_weights
        # bbox_inside_weights = bbox_inside_weights \
        #     .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        # assert bbox_inside_weights.shape[2] == height
        # assert bbox_inside_weights.shape[3] == width
        # outputs.append(torch.from_numpy(bbox_inside_weights))
        # top[2].reshape(*bbox_inside_weights.shape)
        # top[2].data[...] = bbox_inside_weights

        bbox_inside_weights = bbox_inside_weights.view(1, height, width, A*4)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_inside_weights)

        # bbox_outside_weights
        # bbox_outside_weights = bbox_outside_weights \
        #     .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        # assert bbox_outside_weights.shape[2] == height
        # assert bbox_outside_weights.shape[3] == width
        # outputs.append(torch.from_numpy(bbox_outside_weights))
        # top[3].reshape(*bbox_outside_weights.shape)
        # top[3].data[...] = bbox_outside_weights

        bbox_outside_weights = bbox_outside_weights.view(1, height, width, A*4)\
                                .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


# def _unmap(data, count, inds, fill=0):
#     """ Unmap a subset of item (data) back to the original set of items (of
#     size count) """

#     if len(data.shape) == 1:
#         ret = np.empty((count, ), dtype=np.float32)
#         ret.fill(fill)
#         ret[inds] = data
#     else:
#         pdb.set_trace()
#         ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
#         ret.fill(fill)
#         ret[inds, :] = data
#     return ret

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = torch.Tensor(count).fill_(fill).type_as(data)
        ret[inds] = data
    else:
        ret = torch.Tensor(count, data.size(1)).fill_(fill).type_as(data)
        ret[inds,:] = data
    return ret

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.size(0) == gt_rois.size(0)
    assert ex_rois.size(1) == 4
    assert gt_rois.size(1) == 5

    return bbox_transform(ex_rois, gt_rois[:, :4])
