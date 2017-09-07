
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb
    
class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = 20
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    blobs = get_minibatch(minibatch_db, self._num_classes)
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)
    if self.training:
        need_crop = self._roidb[index_ratio]['need_crop']
        np.random.shuffle(blobs['gt_boxes'])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])

        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range
        anchor_idx_leftmost = (np.floor((index) / self.batch_size)) * self.batch_size
        anchor_idx_leftmost = min(int(anchor_idx_leftmost), self.data_size - 1)

        anchor_idx_rightmost = (np.ceil((index + 1) / self.batch_size)) * self.batch_size - 1
        anchor_idx_rightmost = min(int(anchor_idx_rightmost), self.data_size - 1)

        if self.ratio_list[anchor_idx_rightmost] <= 1:
            # this means that data_width < data_height
            ratio = self.ratio_list[anchor_idx_leftmost]
            
            # if the image need to crop.
            trim_size = int(np.floor(data_width / ratio))
            if need_crop and (data_height-trim_size) > 0:
                min_y = torch.min(gt_boxes[:,1])
                max_y = torch.max(gt_boxes[:,3])

                box_region = max_y - min_y + 1
                tmp_y = trim_size - box_region
                # if the bbox region is in the ratio, just random crop.
                if min_y == 0:
                    y_s = 0
                else:
                    if tmp_y > 0:
                        y_s = np.random.randint(np.maximum((max_y-trim_size), 0), 
                                            np.minimum(min_y, data_height-trim_size))
                    elif tmp_y == 0:
                        y_s = int(min_y)
                    else:
                        y_s = int(min_y) + np.random.randint(-tmp_y / 2)
                    
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - y_s
                gt_boxes[:, 3] = gt_boxes[:, 3] - y_s        

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)
                        

            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                             data_width, 3).zero_()

            padding_data[:data_height, :, :] = data[0]

            # update im_info
            im_info[0, 0] = padding_data.size(0)

            # print("height %d %d \n" %(index, anchor_idx))

        elif self.ratio_list[anchor_idx_leftmost] >= 1:
            # this means that data_width > data_height
            ratio = self.ratio_list[anchor_idx_rightmost]

            # if the image need to crop.
            trim_size = int(np.floor(data_height * ratio))
            if need_crop and (data_width-trim_size) > 0:

                min_x = torch.min(gt_boxes[:,0])
                max_x = torch.max(gt_boxes[:,2])
                box_region = max_x - min_x + 1
                tmp_x = trim_size - box_region               
                # if the bbox region is in the ratio, just random crop.
                if min_x == 0:
                    x_s = 0
                else:
                    if tmp_x > 0:            
                        x_s = np.random.randint(np.maximum((max_x-trim_size), 0), 
                                            np.minimum(min_x, data_width-trim_size))
                    elif tmp_x == 0:
                        x_s = int(min_x)
                    else:
                        x_s = int(min_x) + np.random.randint(-tmp_x / 2)

                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - x_s
                gt_boxes[:, 2] = gt_boxes[:, 2] - x_s    
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

            padding_data = torch.FloatTensor(data_height, \
                                             int(np.ceil(data_height * ratio)), 3).zero_()

            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)

        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            gt_boxes.clamp_(0, trim_size)

            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size

        num_boxes = min(gt_boxes.size(0), self.max_num_box)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        # take the top num_boxes
        gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]

        # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)

        return padding_data, im_info, gt_boxes_padding, num_boxes
    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)

        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0

        return data, im_info, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)
