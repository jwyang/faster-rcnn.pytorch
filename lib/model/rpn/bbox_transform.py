# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)
    
    return targets

def bbox_transform_inv(boxes, deltas, batch_size):

    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0]
    dy = deltas[:, :, 1]
    dw = deltas[:, :, 2]
    dh = deltas[:, :, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    #pred_boxes = torch.Tensor(deltas)
    # x1
    #pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    #pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    #pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    #pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    # avoid re-initialize the memory here. 
    pred_boxes = torch.stack([pred_ctr_x - 0.5 * pred_w, 
                        pred_ctr_y - 0.5 * pred_h, 
                        pred_ctr_x + 0.5 * pred_w, 
                        pred_ctr_y + 0.5 * pred_h],1).view(batch_size, -1,4) 

    return pred_boxes

def clip_boxes(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    im_shape = im_shape.int()

    for i in range(batch_size):
        # x1 >= 0
        boxes[i, :, 0].clamp(0, im_shape[i, 1] - 1)
        # boxes[:, 0::4] = torch.max(torch.min(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[i, :, 1].clamp(0, im_shape[i, 0] - 1)
        # boxes[:, 1::4] = torch.max(torch.min(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[i, :, 2].clamp(0, im_shape[i, 1] - 1)
        # boxes[:, 2::4] = torch.max(torch.min(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[i, :, 3].clamp(0, im_shape[i, 0] - 1)
        # boxes[:, 3::4] = torch.max(torch.min(boxes[:, 3::4], im_shape[0] - 1), 0)

    return boxes




def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) * 
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) * 
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4) 
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) - 
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) - 
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps