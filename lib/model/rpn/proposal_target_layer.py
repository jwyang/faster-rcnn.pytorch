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
from ..utils.config import cfg
from bbox_transform import bbox_transform, bbox_overlaps1
from ..utils.cython_bbox import bbox_overlaps
import pdb

DEBUG = False

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.bbox_targets = torch.FloatTensor(1)
        self.bbox_inside_weights = torch.FloatTensor(1)
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

        if cfg.CUDA:
            self.bbox_targets = self.bbox_targets.cuda()
            self.bbox_inside_weights = self.bbox_inside_weights.cuda()
            self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.cuda()
            self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.cuda()
            self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.cuda()


    def forward(self, all_rois, gt_boxes):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        # all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        # gt_boxes = bottom[1].data

        # Include ground-truth boxes in the set of candidate rois
        #all_rois_np = all_rois.numpy()
        #gt_boxes_np = gt_boxes.numpy()

        #zeros = np.zeros((gt_boxes_np.shape[0], 1), dtype=gt_boxes_np.dtype)
        #all_rois_np = np.vstack(
        #    (all_rois_np, np.hstack((zeros, gt_boxes_np[:, :-1])))
        #)

        zeros = torch.Tensor(gt_boxes.size(0), 1).zero_().type_as(all_rois)
        all_rois = torch.cat(
                [all_rois, torch.cat([zeros, gt_boxes[:, :-1]], 1)], 0)

        # Sanity check: single batch only
        # TODO: determined by GPU number
        # assert np.all(all_rois[:, 0] == 0), \
        #         'Only single item batches are supported'

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        # Sample rois with classification labels and bounding box regression
        # targets

        # labels_old, rois_old, bbox_targets, bbox_inside_weights = _sample_rois(
        #     all_rois_np, gt_boxes_np, fg_rois_per_image,
        #     rois_per_image, self._num_classes)

        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)


        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # rois = torch.from_numpy(rois.reshape(-1, 5))
        # labels = torch.from_numpy(labels.reshape(-1, 1))
        # bbox_targets = torch.from_numpy(bbox_targets.reshape(-1, self._num_classes * 4))
        # bbox_inside_weights = torch.from_numpy(bbox_inside_weights.reshape(-1, self._num_classes * 4))
        # bbox_outside_weights = (bbox_inside_weights > 0).float()
        # torch.from_numpy(np.array(bbox_inside_weights > 0).astype(np.float32))

        rois = rois.view(-1, 5)
        labels = labels.view(-1, 1)
        bbox_targets = bbox_targets.view(-1, self._num_classes * 4)
        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): N x 4K blob of regression targets
            bbox_inside_weights (ndarray): N x 4K blob of loss weights
        """

        clss = bbox_target_data[:, 0]
        self.bbox_targets.resize_(clss.size(0), 4 * num_classes).zero_()
        self.bbox_inside_weights.resize_(self.bbox_targets.size()).zero_()
        inds = torch.nonzero(clss > 0).squeeze()

        for i in range(inds.numel()):
            ind = inds[i]
            cls = clss[ind]
            start = int(4 * cls)
            end = start + 4
            self.bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            self.bbox_inside_weights[ind, start:end] = self.BBOX_INSIDE_WEIGHTS

        return self.bbox_targets, self.bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois, labels):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(0) == gt_rois.size(0)
        assert ex_rois.size(1) == 4
        assert gt_rois.size(1) == 4

        targets = bbox_transform(ex_rois, gt_rois)
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return torch.cat([labels.view(-1,1), targets], 1)


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps1(all_rois[:, 1:5].contiguous(),
                                gt_boxes[:, :4].contiguous())

        max_overlaps, gt_assignment = torch.max(overlaps, 1)
        labels = gt_boxes[:,4][gt_assignment]

        # Select foreground RoIs as those with >= FG_THRESH overlap
        # fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
        fg_inds = torch.nonzero(max_overlaps >= cfg.TRAIN.FG_THRESH).squeeze()

        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.numel())
        # Sample foreground regions without replacement
        if fg_inds.numel() > 0:            
            rand_num = torch.randperm(fg_inds.numel()).long()
            if cfg.CUDA:
                rand_num = rand_num.cuda()

            fg_inds = fg_inds[rand_num[:fg_rois_per_image]]

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)

        bg_inds = torch.nonzero((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                                (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)).squeeze()

        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.numel() > 0:
            rand_num = torch.randperm(bg_inds.numel()).long()
            if cfg.CUDA:
                rand_num = rand_num.cuda()

            bg_inds = bg_inds[rand_num[:bg_rois_per_this_image]]

        # The indices that we're selecting (both fg and bg)
        keep_inds = torch.cat([fg_inds, bg_inds], 0)
        # Select sampled values from various arrays:
        labels = labels[keep_inds]
        # Clamp labels for the background RoIs to 0
        labels[fg_rois_per_this_image:] = 0
        rois = all_rois[keep_inds]

        bbox_target_data = self._compute_targets_pytorch(
            rois[:, 1:5], gt_boxes[:,:4][gt_assignment[keep_inds]], labels)

        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels_pytorch(bbox_target_data, num_classes)

        return labels, rois, bbox_targets, bbox_inside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    pdb.set_trace()
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    targets = bbox_transform(torch.from_numpy(ex_rois), torch.from_numpy(gt_rois))
    targets = targets.numpy()
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)



def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
