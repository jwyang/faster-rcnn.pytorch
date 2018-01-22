import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, \
    _affine_grid_gen, _affine_theta
from oim import oim


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, training, query):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.training = training
        self.query = query
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn (query net does not need rpn)
        if not self.query:  # FIXME: maybe here is an error about roi-pooling
            self.RCNN_rpn = _RPN(self.dout_base_model)

            self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
            self.RCNN_roi_pool = _RoIPooling(
                cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
            self.RCNN_roi_align = RoIAlignAvg(
                cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

            self.grid_size = cfg.POOLING_SIZE * 2 \
                if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
            self.RCNN_roi_crop = _RoICrop()

        # TODO: set different num_pid and queue_size for different datasets
        if self.training:
            self.num_pid = 5532
            self.queue_size = 5000
            self.lut_momentum = 0.5

            self.register_buffer('lut', torch.zeros(
                self.num_pid, self.reid_feat_dim).cuda())
            self.register_buffer('queue', torch.zeros(
                self.queue_size, self.reid_feat_dim).cuda())

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data  # Note that gt_boxes is not Variable
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # TODO: maybe query does not need RPN but only roi-pooling
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info,
                                                          gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(
                rois, gt_boxes, num_boxes, self.num_pid)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, \
            aux_label = roi_data

            rois_label = Variable(rois_label.view(-1))
            # add auxiliary_label
            aux_label = Variable(aux_label.view(-1))
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(
                rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            aux_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = \
            #   _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:],
                                       self.grid_size)
            grid_yx = torch.stack(
                [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]],
                3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat,
                                             Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # get the rpn loss.
        rpn_loss = rpn_loss_cls + rpn_loss_bbox

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0),
                                            int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.long().view(
                                                rois_label.size(0), 1,
                                                1).expand(rois_label.size(0),
                                                          1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        # get reid feature, remember to normalize
        reid_feat = F.normalize(self.REID_feat_net(pooled_feat))

        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # set reid loss
        self.REID_loss = 0

        if self.training:
            # classification loss
            label = rois_label.long()
            self.fg_cnt = torch.sum(label.data.ne(0))
            self.bg_cnt = label.data.numel() - self.fg_cnt

            self.RCNN_loss_cls = F.cross_entropy(cls_score, label)

            # bounding box regression L1 loss
            self.RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target,
                                                  rois_inside_ws,
                                                  rois_outside_ws)

            # OIM loss
            # TODO: optimize the algorithm here
            # aux_label is used to update lut and queue
            # pid_label is used to compute loss
            aux_label_np = aux_label.data.cpu().numpy()
            invalid_ind = np.where(
                (aux_label_np < 0) | (aux_label_np >= self.num_pid))
            aux_label_np[invalid_ind] = -1
            pid_label = Variable(
                torch.from_numpy(aux_label_np).long().cuda()).view(-1)
            aux_label = aux_label.long().view(-1)

            reid_result = oim(reid_feat, aux_label, self.lut, self.queue,
                              momentum=self.lut_momentum)
            reid_loss_weight = torch.cat([torch.ones(self.num_pid).cuda(),
                                          torch.zeros(self.queue_size).cuda()])
            self.REID_loss = F.cross_entropy(
                reid_result * 10., pid_label, weight=reid_loss_weight,
                ignore_index=-1)

        rcnn_loss = self.RCNN_loss_cls + self.RCNN_loss_bbox
        reid_loss = self.REID_loss

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss, rcnn_loss, reid_loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        truncated = cfg.TRAIN.TRUNCATED
        if not self.query:
            normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, truncated)
            normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, truncated)
            normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, truncated)
            normal_init(self.RCNN_cls_score, 0, 0.01, truncated)
            normal_init(self.RCNN_bbox_pred, 0, 0.001, truncated)

        # initialize reid net anyway
        normal_init(self.REID_feat_net, 0, 0.01, truncated)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
