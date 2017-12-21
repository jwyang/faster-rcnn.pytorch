# A *Faster* Pytorch Implementation of Faster R-CNN

## Introduction

This project is a *faster* faster R-CNN implementation, aimed to accelerating the training of faster R-CNN object detection models. Recently, there are a number of good implementations:

* [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), developed based on Pycaffe + Numpy

* [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch), developed based on Pytorch + Numpy

* [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), developed based on TensorFlow + Numpy

* [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), developed based on Pytorch + TensorFlow + Numpy

During our implementing, we referred the above implementations, especailly [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch). However, our implementation has several unique and new features compared with the above implementations:

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch.

* **It supports trainig batchsize > 1**. We revise all the layers, including dataloader, rpn, roi-pooling, etc., to train with multiple images at each iteration.

* **It supports multiple GPUs**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

* **It supports three pooling methods**. We integrate three pooling methods: roi pooing, roi align and roi crop. Besides, we convert them to support multi-image batch training.

* **It is memory efficient**. We limit the image aspect ratio, and group the image in batch with similar aspect ratio. We can train resnet101 and VGG16 with batchsize = 4 (4 images) on a sigle Titan X 12 GB. When training with 8 GPU, the maximum batchsize for each GPU is 3 images (Res101), with total batchsize = 24. 

* **It is faster**. Based on the above modifications, the training is faster. We report the training speed on NVIDIA TITAN Xp in the tables below.

## Benchmarking

We benchmark our code thoroughly on three datasets: pascal voc, coco and imagenet-200, using two different network architecture: vgg16 and resnet101. Below are the results:

1). PASCAL VOC 2007 (Train/Test: 07trainval/07test)

model    | GPUs | Batch Size | lr        | lr_decay | max_epoch     |  Speed/epoch | Memory/GPU | mAP 
---------|-----------|----|-----------|-----|-----|-------|--------|--------
VGG-16     | 1 TitanX | 1 | 1e-3 | 5   | 7   |  0.76 hr | 3265MB   | 70.3   
VGG-16     | 1 TitanX | 4 | 4e-3 | 7   | 8   |  0.50 hr | 9083MB   | 70.1   
VGG-16     | 8 TitanX | 16| 1e-2 | 8   | 10  |  0.19 hr | 5291MB   | 69.6 
VGG-16     | 8 TitanX | 24| 1e-2 | 10  | 11  |  0.16 hr | 11303MB  | 69.6   
Res-101    | 1 TitanX | 1 | 1e-3 | 5   | 7   |  0.88 hr | 3200 MB  | 75.0   
Res-101    | 1 TitanX | 4 | 4e-3 | 8   | 10  |  0.60 hr | 9700 MB  | 74.8
Res-101    | 8 TitanX | 16| 1e-2 | 8   | 10  |  0.23 hr | 8400 MB  | 74.4 
Res-101    | 8 TitanX | 24| 1e-2 | 10  | 12  |  0.17 hr | 10327MB  | 74.5   


2). COCO (Train/Test: coco_train/coco_test)

model     | GPUs | Batch Size |lr        | lr_decay | max_epoch     |  Speed/epoch | Memory/GPU | mAP 
---------|-----------|-----|-----------|-----|-----|-------|--------|-----
VGG-16     | 1 TitanX | 1     |1e-3| 5   | 7  |  N/A | N/A   | N/A   
VGG-16     | 1 TitanX | 4     |4e-3| 8   | 10 |  N/A | N/A   | N/A   
VGG-16     | 8 TitanX | 16    |1e-2| 4   | 6  |  N/A | N/A  | N/A 
Res-101    | 1 TitanX | 1     |1e-3| 4   | 6  |  N/A | N/A  | N/A   
Res-101    | 1 TitanX | 4     |4e-3| 4   | 6  |  N/A | N/A  | N/A
Res-101    | 8 TitanX | 16    |1e-2| 4   | 6  |  6.0 hr    |10956 MB  | 36.7  

**NOTE**. N/A means not available now. The benchmarking performance on these datasets will come along with our report soon. Though lack of the benchmark here, you can definitely use the code now! Train your model with the recent code, you will obtain a comparable object detection model to previous implementations on different datasets.

### What we are doing now

* Run systematical experiments on PASCAL VOC 07/12, COCO, ImageNet, Visual Genome (VG) with different settings.

* Find the right training regime for multi-GPU and multi-Image batch training. Now training with 24 images on 8 GPUs has a slight degradation of performance (>1.0 mAP drop on PASCAL VOC.)

* Write a detailed report about the new stuffs in our implementations, and the quantitative results in our experiments.

## Preparation 

First of all, create a folder:
```
mkdir data
```

### Data Preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.

* **COCO**: Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data.

* **Visual Genome**: Please follow the instructions in [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) to prepare Visual Genome dataset. You need to download the images and object annotation files first, and then perform proprecessing to obtain the vocabulary and cleansed annotations based on the scripts provided in this repository.

### Pretrained Model

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

* VGG16: https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0

* ResNet101: https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0

Download them and put them into the data/.

**NOTE**. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results. 

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

### Compilation

Compile the dependencies using following simple commands:

```
cd lib
sh make.sh
```

## Train 

To train a faster R-CNN model with vgg16 on pascal_voc, simply run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py --dataset pascal_voc --net vgg16 --cuda --bs $BATCH_SIZE
```
where 'bs' is the batch size with default 1. Alternatively, to train with resnet101 on pascal_voc, simple run:
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py --dataset pascal_voc --net resnet101 --cuda --bs $BATCH_SIZE
```

Above, BATCH_SIZE can be set adaptively according to your GPU memory size. **On Titan Xp with 12G memory, it can be up to 4**.

If you have multiple (say 8) Titan Xp GPUs, then just use them all! Try:
```
python trainval_net.py --dataset pascal_voc --net vgg16 --cuda --mGPUs --bs 24
```

Change dataset to "coco" or 'vg' if you want to train on COCO or Visual Genome.

## Test

If you want to evlauate the detection performance of a pre-trained vgg16 model on pascal_voc test set, simply run
```
python test_net.py --dataset pascal_voc --net vgg16 --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT --cuda
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.

## Authorship

This project is equally contributed by [Jianwei Yang](https://github.com/jwyang) and [Jiasen Lu](https://github.com/jiasenlu).

## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu, Dhruv Batra, Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    } 
    
    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
