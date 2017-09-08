# Pytorch Faster-RCNN

### Introduction

This project is aimed to reproduce the faster rcnn object detection model. It is developed based on the following projects:

1. [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), developed based on Pycaffe + Numpy

2. [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch), developed based on Pytorch + Numpy

3. [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), developed based on TensorFlow + Numpy

4. [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), developed based on Pytorch + TensorFlow + Numpy

However, there are several unique features compared with the above implementations:

1) **It is pure Pytorch code**. We convert all the numpy implementations to pytorch.

2) **It supports trainig batchsize > 1**. We revise all the layers, including dataloader, rpn, roi-pooling, etc., to train with multiple images at each iteration.

3) **It supports multiple GPUs**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

4) **It is memory efficient**. We limit the image aspect ratio, and group the image in batch with similar aspect ratio. We can train resnet101 and VGG16 with batchsize = 4 (4 images) on a sigle Titan X 12 GB. When training with 8 GPU, the maximum batchsize for each GPU is 3 images (Res101), with total batchsize = 24. 

5) **It is faster**. With above merits, our training speed can achieve xxx / xxx (VGG/Res101) on single TITAN X Pascal GPU and xxx/xxx (VGG / Res101) on 8 TITAN X Pascal GPU.  

### Benchmarking

We benchmark our code thoroughly on three datasets: pascal voc, mscoco and imagenet-200, using two different network architecture: vgg16 and resnet101. Below are the results:

1. PASCAL VOC

	 model     | Train Set | Test Set  | GPUs     | Batch Size |  Speed / epoch | Memory / GPU | mAP 
	-----------|-----------|-----------|----------|------------|-------|--------|-----
	VGG-16     | 07trainval| 07test    |1 Titan X | 1          |  0    | 0      | 0   
	VGG-16     | 07trainval| 07test    |1 Titan X | 4          |  0    | 0      | 0   
	VGG-16     | 07trainval| 07test    |8 Titan X | 27         |  0    | 0      | 0   
	Res-101    | 07trainval| 07test    |1 Titan X | 1          |  0.58 hr | 3200 MB  | 0   
	Res-101    | 07trainval| 07test    |1 Titan X | 4          |  0.48 hr | 9800 MB  | 0   
	Res-101    | 07trainval| 07test    |8 Titan X | 27         |  0.16 hr | 8400 MB     | 0   


1. COCO

	 model     | Train Set | Test Set  | GPUs     | Batch Size | Speed / epoch | Memory / GPU | mAP 
	-----------|-----------|-----------|----------|------------|-------|--------|-----
	VGG-16     | coco_train| coco_test |1 Titan X | 1          |  0    | 0   | 0   
	VGG-16     | coco_train| coco_test |1 Titan X | 4          |  0    | 0   | 0   
	VGG-16     | coco_train| coco_test |8 Titan X | 27         |  0    | 0   | 0   
	Res-101    | coco_train| coco_test |1 Titan X | 1          |  14 hr| 3300 MB | 0   
	Res-101    | coco_train| coco_test |1 Titan X | 4          |  12 hr| 9800 MB | 0   
	Res-101    | coco_train| coco_test |8 Titan X | 27         |  4 hr | 8400 MB | 0  

#### Prepare Data 
**PASCAL_VOC** and **COCO**:

Please follow the instructions of [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup VOC and COCO datasets. The steps involve downloading data and optionally creating softlinks in the data folder. Since faster RCNN does not rely on pre-computed proposals, it is safe to ignore the steps that setup proposals.

**ImageNet**:





To train a resnet101, run:
```
 CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net resnet101
 ```
Alternatively, to train a vgg16, run:
```
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net vgg16
```

