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
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils import network
import time
import pdb
from model.utils.network import _smooth_l1_loss

# from model.utils.vgg16 import VGG16

class _RCNN_base(nn.Module):
    def __init__(self, baseModels, classes):
        super(_RCNN_base, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)

        self.RCNN_base_model = nn.Sequential()
        for i in range(len(baseModels)):
            self.RCNN_base_model.add_module('part{}'.format(i), baseModels[i])

        virtual_input = torch.randn(1, 3, cfg.TRAIN.TRIM_HEIGHT, cfg.TRAIN.TRIM_WIDTH)
        out = self.RCNN_base_model(Variable(virtual_input))
        self.feat_height = out.size(2)
        self.feat_width = out.size(3)
        self.dout_base_model = out.size(1)
        # define rpn
        self.RCNN_rpn = _RPN(self.feat_height, self.feat_width, self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_pool = _RoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

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

        rois_var = Variable(rois.view(-1,5))

        # do roi pooling based on predicted rois

        pooled_feat = self.RCNN_roi_pool(base_feat, rois_var)
        pooled_feat_all = pooled_feat.view(pooled_feat.size(0), -1)

        return rois, pooled_feat_all, rois_label, rois_target, rois_inside_ws, rois_outside_ws, rpn_loss_cls, rpn_loss_bbox

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, baseModel, classes, debug=False):
        super(_fasterRCNN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)

        # define base model, e.g., VGG16, ResNet, etc.
        if baseModel == "vgg16":
            slices = network.load_baseModel(baseModel)
            self.RCNN_base = _RCNN_base(slices[:3], classes)
            self.RCNN_fc6 = slices[3]
            self.RCNN_fc7 = slices[4]
        elif baseModel == "res50":
            pretrained_model = models.resnet50(pretrained=True)
            RCNN_base_model = nn.Sequential(*list(pretrained_model.children())[:-2])
        elif baseModel == "res101":
            pretrained_model = models.resnet50(pretrained=True)
            RCNN_base_model = nn.Sequential(*list(pretrained_model.children())[:-2])
        else:
            raise RuntimeError('baseModel is not included.')

        self.dout_base_model = self.RCNN_base.dout_base_model

        self.RCNN_cls_score = nn.Sequential(
            nn.Linear(4096, self.n_classes)
        )

        self.RCNN_bbox_pred = nn.Sequential(
            nn.Linear(4096, self.n_classes * 4)
        )

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # for log
        self.debug = debug

    def forward(self, im_data, im_info, gt_boxes, num_boxes):


        batch_size = im_data.size(0)
        rois, pooled_feat_all, rois_label, rois_target, rois_inside_ws, rois_outside_ws, \
                rpn_loss_cls, rpn_loss_bbox = self.RCNN_base(im_data, im_info, gt_boxes, num_boxes)

        rpn_loss = rpn_loss_cls + rpn_loss_bbox

        # feed pooled features to top model
        x = self.RCNN_fc6(pooled_feat_all)
        x = F.relu(x, inplace = True)
        x = F.dropout(x, training=self.training)

        x = self.RCNN_fc7(x)
        x = F.relu(x, inplace = True)
        x = F.dropout(x, training=self.training)

        # x = self.RCNN_top_model(pooled_feat_all)

        # compute classifcation loss
        cls_score = self.RCNN_cls_score(x)
        cls_prob = F.softmax(cls_score)

        # pdb.set_trace()

        # compute regression loss
        bbox_pred = self.RCNN_bbox_pred(x)

        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            label = rois_label.long()
            self.fg_cnt = torch.sum(label.data.ne(0))
            self.bg_cnt = label.data.numel() - self.fg_cnt

            ce_weights = rois_label.data.new(cls_score.size(1)).fill_(1)
            ce_weights[0] = float(self.fg_cnt) / self.bg_cnt

            # self.RCNN_loss_cls = F.cross_entropy(cls_score, label, weight=ce_weights)

            self.RCNN_loss_cls = F.cross_entropy(cls_score, label)

            # bounding box regression L1 loss
            # rois_target = torch.mul(rois_target, rois_inside_ws)
            # bbox_pred = torch.mul(bbox_pred, rois_inside_ws)

            # self.RCNN_loss_bbox = F.smooth_l1_loss(bbox_pred, rois_target, size_average=False) / (self.fg_cnt + 1e-4)

            self.RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rcnn_loss = self.RCNN_loss_cls + self.RCNN_loss_bbox

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss, rcnn_loss
