# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn_cascade import _fasterRCNN, _RCNN_base
import pdb

class vgg16(_fasterRCNN):
  def __init__(self, classes):
    _fasterRCNN.__init__(self, classes)    
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512

  def _init_modules(self):

    vgg = models.vgg16()
    state_dict = torch.load(self.model_path)
    vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    vgg.features = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in vgg.features[layer].parameters(): p.requires_grad = False

    self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
    self.RCNN_bbox_pred = nn.Linear(4096, 4)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

