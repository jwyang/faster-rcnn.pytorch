
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
    pdb.set_trace()
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    gt_boxes = torch.from_numpy(blobs['gt_boxes'])
    return data, im_info, gt_boxes

  def __len__(self):
    return len(self._roidb)
