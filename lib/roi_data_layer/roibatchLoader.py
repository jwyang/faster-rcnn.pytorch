
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
import numpy as np
import time
import pdb

class roiLoader(data.Dataset):
  def __init__(self, roidb, num_classes):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = 600
    self.trim_width = 900

  def __getitem__(self, index):

    minibatch_db = [self._roidb[index]]
    blobs = get_minibatch(minibatch_db, self._num_classes)
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    gt_boxes = torch.from_numpy(blobs['gt_boxes'])

    ################################################
    # we trim the image according to the gt_boxes, #
    # to retrain the gt boxes as much as possible  #
    ################################################

    x_left_most,_   = torch.min(gt_boxes[:, 0], 0)
    y_top_most, _   = torch.min(gt_boxes[:, 1], 0)
    x_right_most,_  = torch.max(gt_boxes[:, 2], 0)
    y_bottom_most,_ = torch.max(gt_boxes[:, 3], 0)

    trim_data = torch.FloatTensor(1, self.trim_height, self.trim_width, 3)
    u_height = np.min((self.trim_height, data.size(1)))
    u_width = np.min((self.trim_width, data.size(2)))

    # copy data to trim data
    trim_data[1, :u_height, :u_width, :] = data[1, :u_height, :u_width, :]

    # we trim image a bit to adapt to the trim_height
    if data.size(1) > self.trim_height:
        trim_data[0, :, :data.size(2)]
    # we trim image a bit to adapt to the trim_width
    if data.size(2) > self.trim_width:

    # if exceeds the image width
    if (x_right_most - x_left_most) > self.trim_width:
        print("trim")

    pdb.set_trace()

    return data, im_info, gt_boxes

  def __len__(self):
    return len(self._roidb)
