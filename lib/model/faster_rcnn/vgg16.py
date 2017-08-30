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

  def _init_modules(self):

    self.vgg = models.vgg16()
    self.load_pretrained_cnn()

    self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.vgg.features = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.vgg.features[layer].parameters(): p.requires_grad = False

    self.RCNN_base = _RCNN_base(self.vgg.features, self.classes)

    self.RCNN_top = self.vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
    self.RCNN_bbox_pred = nn.Linear(4096, 4)


  def load_pretrained_cnn(self):
    state_dict = torch.load(self.model_path)

    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})
