
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = 20
    self.training = training
    self.normalize = normalize

  def __getitem__(self, index):

    minibatch_db = [self._roidb[index]]
    blobs = get_minibatch(minibatch_db, self._num_classes)
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    np.random.shuffle(blobs['gt_boxes'])
    gt_boxes = torch.from_numpy(blobs['gt_boxes'])
    data_height, data_width = data.size(1), data.size(2)

    if self.training:
        ##################################################
        # we crop the input image to fixed size randomly #
        ##################################################
        # trim_data = torch.FloatTensor(1, self.trim_height, self.trim_width, 3)        
        if data_height > data_width:
            # if height > width, then crop on height
            # randomly generate an y start point
            # while True:
            # assign score to y axis
            y_score = torch.FloatTensor(data_height).zero_()
            for i in range(gt_boxes.size(0)):
                rg = torch.arange(int(gt_boxes[i, 1]), int(gt_boxes[i, 3]))
                score = -(rg - gt_boxes[i, 1]) * (rg - gt_boxes[i, 3]) / (gt_boxes[i, 3] - gt_boxes[i, 1])**2
                y_score[int(gt_boxes[i, 1]):int(gt_boxes[i, 3])] += score

            # find the inds with maximal score in y_score
            if data_height > self.trim_height:

                ys = torch.arange(0, data_height - self.trim_height, 5).long()
                y_score_cum = torch.FloatTensor(ys.size()).zero_()

                for i in range(ys.size(0)):
                    s = ys[i]
                    y_score_cum[i] = y_score[s:s + self.trim_height].sum()

                _, order = torch.sort(y_score_cum, 0, True)

                ys_ordered = ys[order]
                rand_num = torch.randint(min(5, ys_ordered.size(0)))

                ys = ys_ordered[rand_num]
                ys = min(ys, data_width - self.trim_width)
            else:
                y_s = 0
                
            trim_data = data[:, y_s:(y_s + self.trim_height), :]

            # shift y coordiante of gt_boxes
            gt_boxes[:, 1] = gt_boxes[:, 1] - y_s
            gt_boxes[:, 3] = gt_boxes[:, 3] - y_s

            # update gt bounding box according the trip
            gt_boxes[:, 1].clamp_(0, self.trim_height - 1)
            gt_boxes[:, 3].clamp_(0, self.trim_height - 1)

            # update im_info
            im_info[0, 0] = self.trim_height

        elif data_height <= data_width:
            # if height <= width, then crop on width
            # while True:

            # assign score to y axis
            x_score = torch.FloatTensor(data_width).zero_()
            for i in range(gt_boxes.size(0)):
                rg = torch.arange(int(gt_boxes[i, 0]), int(gt_boxes[i, 2]))
                score = -(rg - gt_boxes[i, 0]) * (rg - gt_boxes[i, 2]) / (gt_boxes[i, 2] - gt_boxes[i, 0])**2
                x_score[int(gt_boxes[i, 0]):int(gt_boxes[i, 2])] += score

            # find the inds with maximal score in y_score
            if data_width > self.trim_width:
                xs = torch.arange(0, data_width - self.trim_width, 5).long()
                x_score_cum = torch.FloatTensor(xs.size()).zero_()

                for i in range(xs.size(0)):
                    s = xs[i]
                    x_score_cum[i] = x_score[s:s + self.trim_width].sum()

                _, order = torch.sort(x_score_cum, 0, True)

                xs_ordered = xs[order]
                rand_num = torch.randint(min(5, xs_ordered.size(0)))

                xs = xs_ordered[rand_num]
                xs = min(xs, data_width - self.trim_width)
            else:
                x_s = 0

            trim_data = data[:, :, x_s:(x_s + self.trim_width), :]

            # shift x coordiante of gt_boxes
            gt_boxes[:, 0] = gt_boxes[:, 0] - x_s
            gt_boxes[:, 2] = gt_boxes[:, 2] - x_s

            # update gt bounding box according the trip
            gt_boxes[:, 0].clamp_(0, self.trim_width - 1)
            gt_boxes[:, 2].clamp_(0, self.trim_width - 1)

            im_info[0, 1] = self.trim_width

        num_boxes = min(gt_boxes.size(0), self.max_num_box)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, 5).zero_()
        # take the top num_boxes
        gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]

        # permute trim_data to adapt to downstream processing
        trim_data = trim_data.permute(0, 3, 1, 2).contiguous().view(3, self.trim_height, self.trim_width)
        im_info = im_info.view(3)

        if self.normalize:
            trim_data = trim_data / 255.0
            trim_data = self.normalize(trim_data)

        return trim_data, im_info, gt_boxes, num_boxes
    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        num_boxes = gt_boxes.size(0)
        im_info = im_info.view(3)

        if self.normalize:
            data = data / 255.0
            data = self.normalize(data)

        return data, im_info, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)
