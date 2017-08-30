# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils import network
from model.utils.network import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint

from model.faster_rcnn.faster_rcnn_cascade import _fasterRCNN
import pdb

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res50, res101, res152',
                    default='vgg16', type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      nargs=argparse.REMAINDER)
  parser.add_argument('--ngpu', dest='ngpu',
                      help='number of gpu',
                      default=1, type=int)


# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=10000, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)

  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit(1)

  args = parser.parse_args()
  return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
use_multiGPU = False

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger('./logs')
  
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  imdb, roidb = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  dataset = roibatchLoader(roidb, imdb.num_classes, training=False,
                        normalize = False)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=0)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.ngpu > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.ngpu > 0:
    cfg.CUDA = True

  # initilize the network here.
  fasterRCNN = _fasterRCNN(args.net, imdb.classes)
  # weights_normal_init(fasterRCNN)
  weights_normal_init(fasterRCNN.RCNN_base.RCNN_rpn.RPN_ConvReLU)
  weights_normal_init(fasterRCNN.RCNN_base.RCNN_rpn.RPN_cls_score)
  weights_normal_init(fasterRCNN.RCNN_base.RCNN_rpn.RPN_bbox_pred)
  weights_normal_init(fasterRCNN.RCNN_cls_score)
  weights_normal_init(fasterRCNN.RCNN_bbox_pred, 0.001)

  params = list(fasterRCNN.parameters())

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam([
      {'params': fasterRCNN.RCNN_base.RCNN_base_model[1].parameters(), 'lr': lr},
      {'params': fasterRCNN.RCNN_base.RCNN_base_model[2].parameters()},
      {'params': fasterRCNN.RCNN_base.RCNN_rpn.parameters()},
      {'params': fasterRCNN.RCNN_fc6.parameters()},
      {'params': fasterRCNN.RCNN_fc7.parameters()},
      {'params': fasterRCNN.RCNN_cls_score.parameters()},
      {'params': fasterRCNN.RCNN_bbox_pred.parameters()},
    ], lr = lr)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD([
      {'params': fasterRCNN.RCNN_base.RCNN_base_model[1].parameters(), 'lr': lr},
      {'params': fasterRCNN.RCNN_base.RCNN_base_model[2].parameters()},
      {'params': fasterRCNN.RCNN_base.RCNN_rpn.parameters()},
      {'params': fasterRCNN.RCNN_fc6.parameters(), 'lr': lr},
      {'params': fasterRCNN.RCNN_fc7.parameters(), 'lr': lr},
      {'params': fasterRCNN.RCNN_cls_score.parameters()},
      {'params': fasterRCNN.RCNN_bbox_pred.parameters()},
    ], lr = lr, momentum=momentum, weight_decay=weight_decay)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("loaded checkpoint %s" % (load_name))

  if use_multiGPU:
    fasterRCNN.RCNN_base = nn.DataParallel(fasterRCNN.RCNN_base)

  if args.ngpu > 0:
    fasterRCNN.cuda()

  loss_temp = 0
  start = time.time()

  data_iter = iter(dataloader)

  aspect_ratio = torch.FloatTensor(train_size).zero_()
  for step in range(train_size):
    data = data_iter.next()
    im_data.data.resize_(data[0].size()).copy_(data[0])
    im_info.data.resize_(data[1].size()).copy_(data[1])
    gt_boxes.data.resize_(data[2].size()).copy_(data[2])
    num_boxes.data.resize_(data[3].size()).copy_(data[3])

    # aspect_ratio = height / width
    aspect_ratio[step] = data[1][0][0] / data[1][0][1]


  pdb.set_trace()

  end = time.time()
  print(end - start)
