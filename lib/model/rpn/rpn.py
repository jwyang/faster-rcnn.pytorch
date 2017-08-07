import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from proposal_layer import _ProposalLayer
from anchor_target_layer import _AnchorTargetLayer

class _RPN(nn.Module):
    """ region proposal network """
    _feat_stride = [16, ]        # TODO: figure out what this means
    _anchor_scales = [8, 16, 32]  # TODO: compare this to offcial pycaffe code
    def __init__(self, din):
        super(_RPN, self).__init__()
        self.din = din or 512  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.feat_stride = cfg.FEAT_STRIDE
        
        # define the convrelu layers processing input feature map
        self.RPN_ConvReLU = nn.Sequential(
                        nn.Conv2d(self.din, 512, 3, 1, 1, bias=True),
                        nn.ReLU(True)
        )

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * 3 * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = Conv2d(512, self.nc_score_out, 1, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * 3 * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = Conv2d(512, self.nc_bbox_out, 1, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self._feat_stride, self.anchor_scales)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self._feat_stride, self._anchor_scales)

        # define classifcation loss bwtween cls_score and ground truth labels
        self.loss_cls = 0

        # define regression loss between bbox_pred and ground truth bboxes
        self.loss_bbox = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, input, im_info, gt_bboxes):
        # input is the feature map

        # return feature map after convrelu layer
        rpn_conv1 = self.RPN_ConvReLU(input)

        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.RPN_proposal((rpn_cls_prob, rpn_bbox_pred,
                                 im_info, cfg_key))

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.RPN_anchor_target(rpn_cls_score, gt_bboxes, im_info)

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
            rpn_label = rpn_data[0].view(-1)

            rpn_keep = rpn_label.data.ne(-1).nonzero().squeeze()
            rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

            fg_cnt = torch.sum(rpn_label.data.ne(0))

            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

            # compute bbox regression loss
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
            rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

            self.rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return rois
