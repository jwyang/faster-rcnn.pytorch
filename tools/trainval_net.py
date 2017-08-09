# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import torch
from torch.autograd import Variable
import numpy as np
import argparse
import pprint
import pdb

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.layer import RoIDataLayer
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.faster_rcnn.faster_rcnn import _fasterRCNN

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--worker', dest='worker',
                      help='number of worker loading data',
                      default=0, type=int)


  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set

  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = False  
  imdb, roidb = combined_roidb(args.imdbval_name)
  
  print('{:d} roidb entries'.format(len(roidb)))
  train_loader = RoIDataLayer(roidb, imdb.num_classes)

  # initilize the network here.
  fasterRCNN = _fasterRCNN(args.net, imdb.classes)

  # training
  for i in range(10):
    blobs = train_loader.forward()
    im_data = Variable(torch.from_numpy(blobs['data']))
    im_info = Variable(torch.from_numpy(blobs['im_info']))
    gt_boxes = Variable(torch.from_numpy(blobs['gt_boxes']))
    # gt_ishard = blobs['gt_ishard']
    # dontcare_areas = blobs['dontcare_areas']
    if gt_boxes.size(0) == 1:
      continue

    out = fasterRCNN(im_data, im_info, gt_boxes)

    pdb.set_trace()
