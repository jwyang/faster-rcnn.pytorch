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

class vgg16():
  def __init__(self):
    self.vgg = models.vgg16()
    # Remove fc8
    # self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.vgg.features[layer].parameters(): p.requires_grad = False

  # def _image_to_head(self):
  #   net_conv = self._layers['head'](self._image)
  #   self._act_summaries['conv']['value'] = net_conv
    
  #   return net_conv

  # def _head_to_tail(self, pool5):
  #   pool5_flat = pool5.view(pool5.size(0), -1)
  #   fc7 = self.vgg.classifier(pool5_flat)

  #   return fc7
  def slice(self):

      self.slices = []
      # we fix conv1_1, conv1_2, conv2_1, conv2_2
      self.slices.append(nn.Sequential(*list(self.vgg.features.children())[:10]))
      # we finetune conv3_1, conv3_2, conv3_3
      self.slices.append(nn.Sequential(*list(self.vgg.features.children())[10:17]))
      # we retrain conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3
      self.slices.append(nn.Sequential(*list(self.vgg.features.children())[17:-1]))

      # we copy fc6
      self.slices.append(self.vgg.classifier[0])

      # we copy fc7
      self.slices.append(self.vgg.classifier[3])

      return self.slices

  def load_pretrained_cnn(self, state_dict):
    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})