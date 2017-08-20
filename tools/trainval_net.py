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
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import pdb

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
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=10, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=100, type=int)  
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      nargs=argparse.REMAINDER)  
  parser.add_argument('--ngpu', dest='ngpu',
                      help='number of gpu',
                      default=1, type=int)


  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
use_multiGPU = True

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

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
  cfg.TRAIN.USE_FLIPPED = True
  imdb, roidb = combined_roidb(args.imdb_name)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  dataset = roibatchLoader(roidb, imdb.num_classes)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=14,
                            shuffle=False, num_workers=4)

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
  weights_normal_init(fasterRCNN.RCNN_rpn.RPN_cls_score)
  weights_normal_init(fasterRCNN.RCNN_rpn.RPN_bbox_pred)  
  weights_normal_init(fasterRCNN.RCNN_top_model)
  weights_normal_init(fasterRCNN.RCNN_cls_score)
  weights_normal_init(fasterRCNN.RCNN_bbox_pred)

  # pdb.set_trace()
  params = list(fasterRCNN.parameters())
  optimizer = optim.Adam(params[8:], lr = lr * 0.1)
  # optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

  if use_multiGPU:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.ngpu > 0:
    fasterRCNN.cuda()

  data_iter = iter(dataloader)
  # training
  # data = data_iter.next()
  # im_data.data.resize_(data[0].size()).copy_(data[0])
  # im_info.data.resize_(data[1].size()).copy_(data[1])
  # gt_boxes.data.resize_(data[2].size()).copy_(data[2])
  # num_boxes.data.resize_(data[3].size()).copy_(data[3])

  loss_temp = 0

  start = time.time()
  for step in range(args.max_iters):

    data = data_iter.next()
    im_data.data.resize_(data[0].size()).copy_(data[0])
    im_info.data.resize_(data[1].size()).copy_(data[1])
    gt_boxes.data.resize_(data[2].size()).copy_(data[2])
    num_boxes.data.resize_(data[3].size()).copy_(data[3])

    fasterRCNN.zero_grad()
    cls_prob, bbox_pred, rpn_loss, rcnn_loss = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
    loss = (rpn_loss.sum() + rcnn_loss.sum()) / im_data.size(0)
    loss_temp += loss.data[0]
    # backward
    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()

    if step % args.disp_interval == 0:
      if use_multiGPU:
        print("[iter %4d] loss: [%.4f] rpn_cls [%.4f] rpn_box [%.4f] rcnn_cls [%.4f] rcnn_box [%.4f] " \
          % (step, loss_temp / 10, 0, 0, 0, 0))      
      else:
        print("[iter %4d] loss: [%.4f] rpn_cls [%.4f] rpn_box [%.4f] rcnn_cls [%.4f] rcnn_box [%.4f] " \
          % (step, loss_temp / 10, fasterRCNN.rpn_loss_cls.data[0], fasterRCNN.rpn_loss_bbox.data[0], \
                                fasterRCNN.RCNN_loss_cls.data[0], fasterRCNN.RCNN_loss_bbox.data[0]))      
      loss_temp = 0

    if (step % args.checkpoint_interval == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        save_net(save_name, fasterRCNN)
        print('save model: {}'.format(save_name))

  end = time.time()
  print(end - start)
