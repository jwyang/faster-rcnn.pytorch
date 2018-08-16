import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.utils.net_utils import _affine_grid_gen
from model.psroi_pooling.modules.psroi_pool import PSRoIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import _smooth_l1_loss

class CoupleNet(nn.Module):
    """ R-FCN """
    def __init__(self, classes, class_agnostic):
        super(CoupleNet, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.box_num_classes = 1 if class_agnostic else self.n_classes

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_crop = _RoICrop()

        self.RCNN_psroi_pool_cls = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
                                          spatial_scale=1/16.0, group_size=cfg.POOLING_SIZE,
                                          output_dim=self.n_classes)
        self.RCNN_psroi_pool_loc = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
                                          spatial_scale=1/16.0, group_size=cfg.POOLING_SIZE,
                                          output_dim=self.box_num_classes * 4)
        self.avg_pooling = nn.AvgPool2d(kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

    def detect_loss(self, cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):
        # classification loss
        RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

        # bounding box regression L1 loss
        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        return RCNN_loss_cls, RCNN_loss_bbox

    def ohem_detect_loss(self, cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):

        def log_sum_exp(x):
            x_max = x.data.max()
            return torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)) + x_max

        num_hard = cfg.TRAIN.BATCH_SIZE * self.batch_size
        pos_idx = rois_label > 0
        num_pos = pos_idx.int().sum()

        # classification loss
        num_classes = cls_score.size(1)
        weight = cls_score.data.new(num_classes).fill_(1.)
        weight[0] = num_pos.data[0] / num_hard

        conf_p = cls_score.detach()
        conf_t = rois_label.detach()

        # rank on cross_entropy loss
        loss_c = log_sum_exp(conf_p) - conf_p.gather(1, conf_t.view(-1,1))
        loss_c[pos_idx] = 100. # include all positive samples
        _, topk_idx = torch.topk(loss_c.view(-1), num_hard)
        loss_cls = F.cross_entropy(cls_score[topk_idx], rois_label[topk_idx], weight=weight)

        # bounding box regression L1 loss
        pos_idx = pos_idx.unsqueeze(1).expand_as(bbox_pred)
        loc_p = bbox_pred[pos_idx].view(-1, 4)
        loc_t = rois_target[pos_idx].view(-1, 4)
        loss_box = F.smooth_l1_loss(loc_p, loc_t)

        return loss_cls, loss_box

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        self.batch_size = im_data.size(0)

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
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

        rois = Variable(rois)

        # Base feature
        base_feat = self.RCNN_conv_new(base_feat)

        # Local feature with PS-ROIPooling
        # Local classification
        local_cls_feat = self.RCNN_local_cls_base(base_feat)
        local_cls_feat = self.RCNN_psroi_pool_cls(local_cls_feat, rois.view(-1, 5))
        local_cls = self.avg_pooling(local_cls_feat)
        local_cls = self.RCNN_local_cls_fc(local_cls)

        # Local bbox regression
        local_bbox_feat = self.RCNN_local_bbox_base(base_feat)
        local_bbox_feat = self.RCNN_psroi_pool_loc(local_bbox_feat, rois.view(-1, 5))
        local_bbox = self.avg_pooling(local_bbox_feat)

        # Global feature with ROIPooling
        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        global_base = self.RCNN_global_base(pooled_feat)
        global_cls = self.RCNN_global_cls(global_base)
        global_bbox = self.RCNN_global_bbox(global_base)

        # fusion global feature and local feature
        cls_score = (local_cls + global_cls).squeeze()
        bbox_pred = (local_bbox + global_bbox).squeeze()

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        cls_prob = F.softmax(cls_score, dim=1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            loss_func = self.ohem_detect_loss if cfg.TRAIN.OHEM else self.detect_loss
            RCNN_loss_cls, RCNN_loss_bbox = loss_func(cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            children_list = list(m.children())
            if len(children_list) > 0:
                m = children_list
            else:
                m = [m]

            for _m in m:
                if isinstance(_m, nn.Conv2d):
                    if truncated:
                        _m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
                    else:
                        _m.weight.data.normal_(mean, stddev)
                        if _m.bias is not None:
                            _m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_conv_1x1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_local_bbox_base, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_local_cls_base, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_local_cls_fc, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_global_base, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_global_cls, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_global_bbox, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
