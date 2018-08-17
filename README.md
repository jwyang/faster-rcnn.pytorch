# A Pytorch Implementation of R-FCN/CoupleNet

## Introduction

This project is an pytorch implement R-FCN and CoupleNet, large part code is reference from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). The R-FCN structure is refer to [Caffe R-FCN](https://github.com/daijifeng001/R-FCN) and [Py-R-FCN](https://github.com/YuwenXiong/py-R-FCN)

- For R-FCN, mAP@0.5 reached 73.2 in VOC2007 trainval dataset
- For CoupleNet, mAP@0.5 reached 75.2 in VOC2007 trainval dataset

## R-FCN

arXiv:1605.06409: [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409)

![15063403082127](http://oodo7tmt3.bkt.clouddn.com/blog_201807132042010817.jpg)

This repo has following modification compare to [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch):

- **R-FCN architecture**: We refered to the origin [Caffe version] of R-FCN, the main structure of R-FCN is show in following figure.
- **PS-RoIPooling with CUDA** :(refer to the other pytorch implement R-FCN, pytorch_RFCN). I have modified it to fit multi-image training (not only batch-size=1 is supported)
- **Implement multi-scale training:** As the original paper says, each image is randomly reized to differenct resolutions (400, 500, 600, 700, 800) when training, and during test time, we use fix input size(600). These make 1.2 mAP gain in our experiments.
- **Implement OHEM:** in this repo, we implement Online Hard Example Mining(OHEM) method in the paper, set `OHEM: False` in `cfgs/res101.yml` for using OHEM. Unluckly, it cause a bit performance degration in my experiments

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180817160334.jpg)

## CoupleNet

arXiv:1708.02863:[CoupleNet: Coupling Global Structure with Local Parts for Object Detection](https://arxiv.org/abs/1708.02863)

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180816205255.png)

- Making changes based on R-FCN
- Implement local/global FCN in CoupleNet

## Tutorial

* [R-FCN blog](http://blog.prince2015.club/2018/07/13/R-FCN/)

## Benchmarking

We benchmark our code thoroughly on three datasets: pascal voc using two different architecture: R-FCN and CoupleNet. Results shows following:

1). PASCAL VOC 2007 (Train: 07_trainval - Test: 07_test, scale=400, 500, 600, 700, 800)

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
R-FCN  | 1 | 2 | 4e-3 | 8   | 7   |  0.88 hr | 3000 MB  | 73.2
CouleNet  | 1 | 2 | 4e-3 | 8   | 10  |  0.60 hr | 8900 MB  | 75.2


## Preparation


First of all, clone the code
```
$ git clone https://github.com/princewang1994/R-FCN.pytorch.git
```

Then, create a folder:
```
$ cd R-FCN.pytorch && mkdir data
$ ln -s $VOC_DEVKIT_ROOT .
```

### prerequisites

* Python 3.6
* Pytorch 0.3.0 or higher
* CUDA 8.0 or higher

### Data Preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.


### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

| GPU model  | Architecture |
| ------------- | ------------- |
| TitanX (Maxwell/Pascal) | sm_52 |
| GTX 960M | sm_50 |
| GTX 1080 (Ti) | sm_61 |
| Grid K520 (AWS g2.2xlarge) | sm_30 |
| Tesla K80 (AWS p2.xlarge) | sm_37 |

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
$ pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
$ cd lib
$ sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

## Train

To train a faster R-CNN model with vgg16 on pascal_voc, simply run:
```
$ CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
				   --arch rfcn \
                   --dataset pascal_voc --net res101 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda
```

- Set `--s` to identified differenct experiments. 
- For CoupleNet training, replace `--arch rfcn` with `--arch couplenet`, other arguments should be modified according to your machine. (e.g. larger learning rate for big batch-size)
- Model are saved to `$RFCN_ROOT/save` 

## Test

If you want to evlauate the detection performance of a pre-trained model on pascal_voc test set, simply run
```
$ python test_net.py --dataset pascal_voc --arch rfcn \
				   --net res101 \
                   --checksession $SESSION \
                   --checkepoch $EPOCH \
                   --checkpoint $CHECKPOINT \
                   --cuda
```
- Specify the specific model session(`--s` in training phase), chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.

## Demo

Below are some detection results:

<div style="color:#0000FF" align="center">
<img src="images/img3_det_res101.jpg" width="430"/> <img src="images/img4_det_res101.jpg" width="430"/>
</div>

## Going to do

- Keeping update structures to reach the state-of-art
- More benchmarking in VOC0712/COCO
- Pretrained model for VOC

## Acknowledgement

This project is equally writen by [Prince Wang](https://github.com/princewang1994), but thanks the faster-rcnn.pytorch's code provider [jwyang](https://github.com/jwyang)
