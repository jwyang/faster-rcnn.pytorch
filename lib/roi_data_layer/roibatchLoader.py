
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
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, num_classes):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH

  def __getitem__(self, index):

    minibatch_db = [self._roidb[index]]
    blobs = get_minibatch(minibatch_db, self._num_classes)
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    gt_boxes = torch.from_numpy(blobs['gt_boxes'])

    ##################################################
    # we crop the input image to fixed size randomly #
    ##################################################

    data_height, data_width = data.size(1), data.size(2)
    # trim_data = torch.FloatTensor(1, self.trim_height, self.trim_width, 3)

    if data_height > data_width:
        # if height > width, then crop on height
        # randomly generate an y start point
        y_s = np.random.randint(data_height - self.trim_height + 1)
        trim_data = data[:, y_s:(y_s + self.trim_height), :]

        # shift y coordiante of gt_boxes
        gt_boxes[:, 1] = gt_boxes[:, 1] - y_s
        gt_boxes[:, 3] = gt_boxes[:, 3] - y_s        

        # update gt bounding box according the trip
        gt_boxes[:, 1].clamp(0, self.trim_height)
        gt_boxes[:, 3].clamp(0, self.trim_height)

        # update im_info
        im_info[0, 0] = self.trim_height

    elif data_height <= data_width:
        # if height <= width, then crop on width
        x_s = np.random.randint(data_width - self.trim_width + 1)
        trim_data = data[:, :, x_s:(x_s + self.trim_width), :]

        # shift x coordiante of gt_boxes
        gt_boxes[:, 0] = gt_boxes[:, 0] - x_s
        gt_boxes[:, 2] = gt_boxes[:, 2] - x_s

        # update gt bounding box according the trip
        gt_boxes[:, 0].clamp(0, self.trim_width)
        gt_boxes[:, 2].clamp(0, self.trim_width)

        # update im_info
        im_info[0, 1] = self.trim_width

    # append img index to im_info and gt_boxes
    ind = torch.FloatTensor(gt_boxes.size(0), 1).fill_(index)
    im_info = torch.cat((im_info, ind[0]), 1)
    gt_boxes = torch.cat((gt_boxes, ind), 1)

    # permute trim_data to adapt to downstream processing
    trim_data = trim_data.permute(0, 3, 1, 2)
    
    return trim_data, im_info, gt_boxes

  def __len__(self):
    return len(self._roidb)
