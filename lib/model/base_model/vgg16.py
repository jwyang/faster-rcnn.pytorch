import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.blob import im_list_to_blob
from network import Conv2d
import network


class _VGG16(nn.Module):
    def __init__(self, bn=False):
        super(VGG16, self).__init__()

        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        network.set_trainable(self.conv1, requires_grad=False)
        network.set_trainable(self.conv2, requires_grad=False)

        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn))

    def forward(self, im_data):
        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        own_dict = self.state_dict()
        for name, val in own_dict.items():
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}/{}:0'.format(i, j, ptype)
            param = torch.from_numpy(params[key])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)


if __name__ == '__main__':
    vgg = _VGG16()
    vgg.load_from_npy_file('/media/longc/Data/models/VGG_imagenet.npy')
