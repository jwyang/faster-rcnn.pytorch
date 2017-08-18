# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
import pdb

from ..utils.cython_bbox import bbox_overlaps, bbox_intersections

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = False


def proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, _num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    gt_ishard: (G, 1) {0 | 1} 1 indicates hard
    dontcare_areas: (D, 4) [ x1, y1, x2, y2]
    _num_classes
    ----------
    Returns
    ----------
    rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
    bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
    bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
    bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format

    # Include ground-truth boxes in the set of candidate rois
    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
        assert gt_ishard.shape[0] == gt_boxes.shape[0]
        gt_ishard = gt_ishard.astype(int)
        gt_easyboxes = gt_boxes[gt_ishard != 1, :]
    else:
        gt_easyboxes = gt_boxes

    """
    add the ground-truth to rois will cause zero loss! not good for visuallization
    """
    jittered_gt_boxes = _jitter_gt_boxes(gt_easyboxes)
    zeros = np.zeros((gt_easyboxes.shape[0] * 2, 1), dtype=gt_easyboxes.dtype)
    all_rois = np.vstack((all_rois, \
                          np.hstack((zeros, np.vstack((gt_easyboxes[:, :-1], jittered_gt_boxes[:, :-1]))))))

    # Sanity check: single batch only
    assert np.all(all_rois[:, 0] == 0), \
        'Only single item batches are supported'

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, gt_boxes, gt_ishard, dontcare_areas, fg_rois_per_image,
        rois_per_image, _num_classes)

    _count = 1
    if DEBUG:
        if _count == 1:
            _fg_num, _bg_num = 0, 0
        print 'num fg: {}'.format((labels > 0).sum())
        print 'num bg: {}'.format((labels == 0).sum())
        _count += 1
        _fg_num += (labels > 0).sum()
        _bg_num += (labels == 0).sum()
        print 'num fg avg: {}'.format(_fg_num / _count)
        print 'num bg avg: {}'.format(_bg_num / _count)
        print 'ratio: {:.3f}'.format(float(_fg_num) / float(_bg_num))

    rois = rois.reshape(-1, 5)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)

    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _sample_rois(all_rois, gt_boxes, gt_ishard, dontcare_areas, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: R x G
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)  # R
    max_overlaps = overlaps.max(axis=1)  # R
    labels = gt_boxes[gt_assignment, 4]

    # preclude hard samples
    ignore_inds = np.empty(shape=(0), dtype=int)
    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
        gt_ishard = gt_ishard.astype(int)
        gt_hardboxes = gt_boxes[gt_ishard == 1, :]
        if gt_hardboxes.shape[0] > 0:
            # R x H
            hard_overlaps = bbox_overlaps(
                np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
                np.ascontiguousarray(gt_hardboxes[:, :4], dtype=np.float))
            hard_max_overlaps = hard_overlaps.max(axis=1)  # R x 1
            # hard_gt_assignment = hard_overlaps.argmax(axis=0)  # H
            ignore_inds = np.append(ignore_inds,
                                    np.where(hard_max_overlaps >= cfg.TRAIN.FG_THRESH)[0])
            if DEBUG:
                if ignore_inds.size > 1:
                    print 'num hard: {:d}:'.format(ignore_inds.size)
                    print 'hard box:', gt_hardboxes
                    print 'rois: '
                    print all_rois[ignore_inds]

    # preclude dontcare areas
    if dontcare_areas is not None and dontcare_areas.shape[0] > 0:
        # intersec shape is D x R
        intersecs = bbox_intersections(
            np.ascontiguousarray(dontcare_areas, dtype=np.float),  # D x 4
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float)  # R x 4
        )
        intersecs_sum = intersecs.sum(axis=0)  # R x 1
        ignore_inds = np.append(ignore_inds,
                                np.where(intersecs_sum > cfg.TRAIN.DONTCARE_AREA_INTERSECTION_HI)[0])
        # if ignore_inds.size >= 1:
        #     print 'num dontcare: {:d}:'.format(ignore_inds.size)
        #     print 'dontcare box:', dontcare_areas.astype(int)
        #     print 'rois: '
        #     print all_rois[ignore_inds].astype(int)

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    fg_inds = np.setdiff1d(fg_inds, ignore_inds)
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    bg_inds = np.setdiff1d(bg_inds, ignore_inds)
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

    # bbox_target_data (1 x H x W x A, 5)
    # bbox_targets <- (1 x H x W x A, K x 4)
    # bbox_inside_weights <- (1 x H x W x A, K x 4)
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

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

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _jitter_gt_boxes(gt_boxes, jitter=0.05):
    """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    """
    jittered_boxes = gt_boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes
