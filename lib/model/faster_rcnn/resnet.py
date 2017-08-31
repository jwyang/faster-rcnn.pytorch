# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.faster_rcnn.faster_rcnn_cascade import _fasterRCNN, _RCNN_base
from model.utils.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import pdb

class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101):
    _fasterRCNN.__init__(self, classes)    
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024

  def _init_modules(self):
    self.resnet = models.resnet101()
    self.load_pretrained_cnn()

    # Fix blocks 
    for p in self.resnet.bn1.parameters(): p.requires_grad=False
    for p in self.resnet.conv1.parameters(): p.requires_grad=False
    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.resnet.layer3.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.resnet.layer2.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.resnet.layer1.parameters(): p.requires_grad=False

    # remove last two layers
    self.resnet.fc = None
    self.resnet.avgpool = None

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.resnet.apply(set_bn_fix)

    # Build resnet.
    self.base_net = nn.Sequential(self.resnet.conv1, self.resnet.bn1,self.resnet.relu, 
      self.resnet.maxpool,self.resnet.layer1,self.resnet.layer2,self.resnet.layer3)

    self.RCNN_base = _RCNN_base(self.base_net, self.classes, self.dout_base_model)

    self.RCNN_top = nn.Sequential(self.resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    self.RCNN_bbox_pred = nn.Linear(2048, 4)


  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.resnet.eval()
      self.resnet.layer1.train()
      if cfg.RESNET.FIXED_BLOCKS >= 1:
        self.resnet.layer2.train()
      if cfg.RESNET.FIXED_BLOCKS >= 2:
        self.resnet.layer3.train()
      if cfg.RESNET.FIXED_BLOCKS >= 3:
        self.resnet.layer4.train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.resnet.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)

    return fc7

  def load_pretrained_cnn(self):
    state_dict = torch.load(self.model_path)
    self.resnet.load_state_dict({k:v for k,v in state_dict.items() if k in self.resnet.state_dict()})
