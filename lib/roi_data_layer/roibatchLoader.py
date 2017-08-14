
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

class roiLoader(data.Dataset):
  def __init__(self, roidb, num_classes, imgsize):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = imgsize
    self.trim_width = imgsize

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
    trim_data = torch.FloatTensor(1, self.trim_height, self.trim_width, 3)

    x_left_most,_   = torch.min(gt_boxes[:, 0], 0)
    y_top_most, _   = torch.min(gt_boxes[:, 1], 0)
    x_right_most,_  = torch.max(gt_boxes[:, 2], 0)
    y_bottom_most,_ = torch.max(gt_boxes[:, 3], 0)

    if data_height > data_width:
        # if height > width, then crop on height
        # randomly generate an y start point
        y_s = np.random.randint(data_height - self.trim_height)
        trim_data = data[:, y_s:(y_s + self.trim_height), :, :]

        # find the bound box outside the image crop
        keep = gt_boxes[:, 3] > y_s and \
               gt_boxes[:, 1] < (y_s + self.trim_height)
        gt_boxes = gt_boxes[keep, :]

        # update gt bounding box according the trip
        gt_boxes = clip_boxes(gt_boxes, [self.trim_height, self.trim_width])

    elif data_height <= data_width:
        # if height <= width, then crop on width
        x_s = np.random.randint(data_width - self.trim_width)
        trim_data = data[:, :, x_s:(x_s + self.trim_width), :, :]
        # find the bound box outside the image crop
        keep = gt_boxes[:, 2] > y_s and \
               gt_boxes[:, 0] < (y_s + self.trim_width)
        gt_boxes = gt_boxes[keep, :]

        # update gt bounding box according the trip
        gt_boxes = clip_boxes(gt_boxes, [self.trim_height, self.trim_width])

    trim_data.permute(0, 3, 1, 2)
    pdb.set_trace()

    return data, im_info, gt_boxes

  def __len__(self):
    return len(self._roidb)
