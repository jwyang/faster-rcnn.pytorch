import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
import numpy as np

from model.utils.config import cfg

from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_pooling_single.modules.roi_pool import _RoIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils import network
import time
import pdb
from model.utils.network import _smooth_l1_loss

# from model.utils.vgg16 import VGG16

class _RCNN_base(nn.Module):
    def __init__(self, baseModels, classes, dout_base_model):
        super(_RCNN_base, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)

        self.RCNN_base_model = baseModels

        # define rpn
        self.RCNN_rpn = _RPN(dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        batch_size = im_data.size(0)
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base_model(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:

            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois = Variable(rois)
            rois_label = Variable(rois_label.view(-1))
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # do roi pooling based on predicted rois
        pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
        # pooled_feat_all = pooled_feat.view(pooled_feat.size(0), -1)

        return rois, pooled_feat, rois_label, rois_target, rois_inside_ws, rois_outside_ws, rpn_loss_cls, rpn_loss_bbox

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_base.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_base.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_base.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):

        batch_size = im_data.size(0)
        rois, feat_out, rois_label, rois_target, rois_inside_ws, rois_outside_ws, \
                rpn_loss_cls, rpn_loss_bbox = self.RCNN_base(im_data, im_info, gt_boxes, num_boxes)

        # get the rpn loss.
        rpn_loss = rpn_loss_cls + rpn_loss_bbox

        # feed pooled features to top model
        feat_out = self._head_to_tail(feat_out)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(feat_out)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(feat_out)
        cls_prob = F.softmax(cls_score)


        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            label = rois_label.long()
            self.fg_cnt = torch.sum(label.data.ne(0))
            self.bg_cnt = label.data.numel() - self.fg_cnt

            self.RCNN_loss_cls = F.cross_entropy(cls_score, label)

            # bounding box regression L1 loss
            self.RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)            

        rcnn_loss = self.RCNN_loss_cls + self.RCNN_loss_bbox

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss, rcnn_loss