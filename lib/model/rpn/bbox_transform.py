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

def bbox_transform_batch(ex_rois, gt_rois):

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
    gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
    gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
    targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)
    
    return targets

def bbox_transform_batch2(ex_rois, gt_rois):

    ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
    ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
    ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
    gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
    gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)
    
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

    # avoid re-initialize the memory here. 
    pred_boxes = torch.stack([pred_ctr_x - 0.5 * pred_w, 
                        pred_ctr_y - 0.5 * pred_h, 
                        pred_ctr_x + 0.5 * pred_w, 
                        pred_ctr_y + 0.5 * pred_h], 2)

    return pred_boxes

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)    

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

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    batch_size = gt_boxes.size(0)
    N = anchors.size(0)
    K = gt_boxes.size(1)

    anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
    gt_boxes = gt_boxes[:,:,:4].contiguous()


    gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
    gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
    gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)    

    anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
    anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
    anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

    boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4) 
    query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

    iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) - 
        torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) - 
        torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
    ih[ih < 0] = 0
    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    # mask the overlap here. 
    overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
    overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), 0)

    return overlaps


def bbox_overlaps_batch2(anchors, gt_boxes):
    """
    anchors: (b, N, 5) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (b, N, K) ndarray of overlap between boxes and query_boxes
    """

    batch_size = gt_boxes.size(0)
    N = anchors.size(1)
    K = gt_boxes.size(1)

    anchors = anchors[:,:,1:5].contiguous()
    gt_boxes = gt_boxes[:,:,:4].contiguous()

    gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
    gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
    gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)    

    anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
    anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
    anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

    boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4) 
    query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

    iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) - 
        torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) - 
        torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
    ih[ih < 0] = 0
    ua = anchors_area + gt_boxes_area - (iw * ih)

    overlaps = iw * ih / ua

    # mask the overlap here. 
    overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), -1)
    overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    return overlaps, anchors_area_zero, gt_area_zero
