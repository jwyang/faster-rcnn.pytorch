import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
import numpy as np

from model.utils.config import cfg
from model.base_model.vgg16 import _VGG16
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.rpn.proposal_target_layer import _ProposalTargetLayer

import pdb

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    n_classes = 21
    classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (600,)
    MAX_SIZE = 1000

    def __init__(self, baseModel, classes=None, debug=False):
        super(_fasterRCNN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)

        # define base model, e.g., VGG16, ResNet, etc.
        if baseModel == "vgg16":
            pretrained_model = models.vgg16(pretrained=True)
            self.RCNN_base_model = nn.Sequential(*list(pretrained_model.features.children())[:-1])

        virtual_input = torch.randn(1, 3, 224, 224)
        out = self.RCNN_base_model(Variable(virtual_input))
        self.dout_base_model = out.size(1)

        # define rpn
        self.RCNN_rpn = _RPN(out.size(1))

        # define proposal layer for target
        self.RPN_proposal_target = _ProposalTargetLayer(self.n_classes)


        self.RCNN_roi_pool = _RoIPooling(7, 7, 1.0/16)

        self.RCNN_top_model = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096)
        )

        self.RCNN_cls_score = nn.Linear(4096, self.n_classes, relu=False)
        self.RCNN_bbox_pred = nn.Linear(4096, self.n_classes * 4, relu=False)

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # for log
        self.debug = debug

    def forward(self, im_data, im_info, gt_boxes=None):

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base_model(im_data)

        # feed base feature map tp RPN to obtain rois
        rois = self.RCNN_rpn(base_feat, im_info, gt_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RPN_proposal_target(rois, gt_boxes)
            rois = roi_data[0]

        # do roi pooling based on predicted rois
        pooled_feat = self.RCNN_roi_pool(base_feat, rois)
        pooled_feat_v = pooled_feat.view(pooled_feat.size()[0], -1)

        # feed pooled features to top model
        x = self.RCNN_top_model(pooled_feat_v)

        # compute classifcation loss
        cls_score = self.RCNN_cls_score(x)
        cls_prob = F.softmax(cls_score)

        # compute regression loss
        bbox_pred = self.RCNN_bbox_pred(x)

        if self.training:
            # classification loss
            label = roi_data[1].squeeze()
            fg_cnt = torch.sum(label.data.ne(0))
            bg_cnt = label.data.numel() - fg_cnt

            # for log
            if self.debug:
                maxv, predict = cls_score.data.max(1)
                self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
                self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
                self.fg_cnt = fg_cnt
                self.bg_cnt = bg_cnt

            ce_weights = torch.ones(cls_score.size()[1])
            ce_weights[0] = float(fg_cnt) / bg_cnt
            ce_weights = ce_weights.cuda()
            self.RCNN_loss_cls = F.cross_entropy(cls_score, label, weight=ce_weights)

            # bounding box regression L1 loss
            bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
            bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
            bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

            self.RCNN_loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return cls_prob, bbox_pred, rois
