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

import torchvision.transforms as transforms

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils import network

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
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='vgg16', type=str)

  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
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
                      default=4, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=4, type=int)

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
                      default=0, type=int)

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
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
                m.bias.data.zero_()

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
use_multiGPU = False

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

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def save_checkpoint(state, filename):
    torch.save(state, filename)

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
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  dataset = roibatchLoader(roidb, imdb.num_classes, training=False,
                        normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]))

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
  weights_normal_init(fasterRCNN.RCNN_bbox_pred)

  params = list(fasterRCNN.parameters())

  if args.optimizer == "adam":
    # lr4ft = lr * 0.01
    # optimizer4ft = torch.optim.Adam(params[4:8], lr = lr4ft)
    # lr4tr = lr * 0.1
    # optimizer4tr = torch.optim.Adam(params[8:], lr = lr4tr)
    lr = lr * 0.1
    optimizer = torch.optim.Adam([
      {'params': fasterRCNN.RCNN_base.RCNN_base_model[0].parameters(), 'lr': lr * 0.0},
      {'params': fasterRCNN.RCNN_base.RCNN_base_model[1].parameters(), 'lr': lr * 0.1},
      {'params': fasterRCNN.RCNN_base.RCNN_base_model[2].parameters()},
      {'params': fasterRCNN.RCNN_base.RCNN_rpn.parameters()},
      {'params': fasterRCNN.RCNN_fc6.parameters()},
      {'params': fasterRCNN.RCNN_fc7.parameters()},
      {'params': fasterRCNN.RCNN_cls_score.parameters()},
      {'params': fasterRCNN.RCNN_bbox_pred.parameters()},
    ], lr = lr)

  elif args.optimizer == "sgd":
    # lr4ft = lr * 0.1
    # optimizer4ft = torch.optim.SGD(params[4:8], lr=lr4ft, momentum=momentum, weight_decay=weight_decay)
    # lr4tr = lr
    # optimizer4tr = torch.optim.SGD(params[8:], lr=lr4tr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.SGD([
      {'params': fasterRCNN.RCNN_base.RCNN_base_model[0].parameters(), 'lr': lr * 0.0},
      {'params': fasterRCNN.RCNN_base.RCNN_base_model[1].parameters(), 'lr': lr * 0.1},
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
    fasterRCNN = load_state_dict(checkpoint['model'])
    optimizer = load_state_dict(checkpoint['optimizer'])
    # optimizer4ft.load_state_dict(checkpoint['optimizer4ft'])
    # optimizer4tr.load_state_dict(checkpoint['optimizer4tr'])
    print("loaded checkpoint %s" % (load_name))

  if use_multiGPU:
    fasterRCNN.RCNN_base = nn.DataParallel(fasterRCNN.RCNN_base)

  if args.ngpu > 0:
    fasterRCNN.cuda()

  for epoch in range(args.start_epoch, args.max_epochs):
    loss_temp = 0
    start = time.time()

    data_iter = iter(dataloader)

    for step in range(train_size):

      data = data_iter.next()

      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      fasterRCNN.zero_grad()
      _, cls_prob, bbox_pred, rpn_loss, rcnn_loss = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      loss = (rpn_loss.sum() + rcnn_loss.sum()) / rpn_loss.size(0)
      loss_temp += loss.data[0]

      # backward
      # optimizer4tr.zero_grad()
      # optimizer4ft.zero_grad()

      optimizer.zero_grad()
      loss.backward()
      network.clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      # optimizer4tr.step()
      # optimizer4ft.step()
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch + 1,
        'model': fasterRCNN.state_dict(),
        "optimizer": optimizer.state_dict(),
      }, save_name)
      print('save model: {}'.format(save_name))

      if step % args.disp_interval == 0:
        if use_multiGPU:
          print("[session %d][epoch %2d][iter %4d] loss: %.4f, lr4ft: %.2e, lr4tr: %.2e" \
            % (args.session, epoch, step, loss_temp / args.disp_interval, lr * 0.1, lr))
          print("\t\t\tfg/bg=(%d/%d)" % (0, 0))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (0, 0, 0, 0))
        else:
          print("[session %d][epoch %2d][iter %4d] loss: %.4f, lr4ft: %.2e, lr4tr: %.2e" \
            % (args.session, epoch, step, loss_temp / args.disp_interval, lr * 0.1, lr))
          print("\t\t\tfg/bg=(%d/%d)" % (fasterRCNN.fg_cnt, fasterRCNN.bg_cnt))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f" %
            (fasterRCNN.RCNN_base.RCNN_rpn.rpn_loss_cls.data[0], \
             fasterRCNN.RCNN_base.RCNN_rpn.rpn_loss_box.data[0], \
             fasterRCNN.RCNN_loss_cls.data[0], \
             fasterRCNN.RCNN_loss_bbox.data[0]))

        loss_temp = 0

      if (step % args.checkpoint_interval == 0) and step > 0:
        #   pdb.set_trace()
          save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
          save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.state_dict(),
            "optimizer": optimizer.state_dict(),
          }, save_name)
          print('save model: {}'.format(save_name))
        #   torch.save(fasterRCNN, save_name)

    if epoch % args.lr_decay_step == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    #   lr4ft = adjust_learning_rate(optimizer4ft)
    #   lr4tr = adjust_learning_rate(optimizer4tr)

    end = time.time()
    print(end - start)
