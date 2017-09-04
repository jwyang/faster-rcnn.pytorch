# Pytorch Faster-RCNN

### Introduction

This project is aimed to reproduce the faster rcnn object detection model. It is developed based on the following projects:

1. [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), developed based on Pycaffe + Numpy

2. [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch), developed based on Pytorch + Numpy

3. [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), developed based on TensorFlow + Numpy

4. [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), developed based on Pytorch + TensorFlow + Numpy

However, there are several unique features compared with the above implementations:

1) **It is pure Pytorch code**. We converted all the numpy implementations to pytorch.

2) **It supports trainig batchsize > 1**. We revised all the layers, including dataloader, rpn, roi-pooling, etc., to train with multiple images at each iteration.

3) **It supports multiple GPUs**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

### Modules

#### Prepare Data

put VOCdevkit2007 under data folder. 

To train a resnet101, run:
```
 CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net resnet101
 ```
Alternatively, to train a vgg16, run:
```
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net vgg16
```

