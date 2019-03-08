#Steps I had to take to recreate

* copy the pretrained .pth model res101 and place in `data/pretrained_model/.`

* simlink to the data downloaded already, via the provided instructions

` ln -s /media/latest_mount/Dev/image_pool/VOC2007/VOCdevkit ./VOCdevkit2007`



* Command given:

```
$ CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net res101 --bs 1 --nw 1 --lr .001 --lr_decay_step 10 --cuda
```

Results took 12 hours roughly to run, and gave: 

```
...
[session 1][epoch 20][iter 10000/10022] loss: 0.0866, lr: 1.00e-04
			fg/bg=(32/96), time cost: 32.098727
			rpn_cls: 0.0031, rpn_box: 0.0355, rcnn_cls: 0.0191, rcnn_box 0.0280
save model: models/res101/pascal_voc/faster_rcnn_1_20_10021.pth
```

I tried to test, but accidentally ended up testing the incorrect pre-trained model

```
$ python test_net.py --dataset pascal_voc --net res101 --cuda

```

Out came the following

```
Called with args:
Namespace(cfg_file='cfgs/vgg16.yml', checkepoch=1, checkpoint=10021, checksession=1, class_agnostic=False, cuda=True, dataset='pascal_voc', large_scale=False, load_dir='models', mGPUs=False, net='res101', parallel_type=0, set_cfgs=None, vis=False)
Using config:
{'ANCHOR_RATIOS': [0.5, 1, 2],
 'ANCHOR_SCALES': [8, 16, 32],
 'CROP_RESIZE_WITH_MAX_POOL': False,
 'CUDA': False,
 'DATA_DIR': '/media/latest_mount/Dev/git/jwyang/faster-rcnn.pytorch/data',
 'DEDUP_BOXES': 0.0625,
 'EPS': 1e-14,
 'EXP_DIR': 'res101',
 'FEAT_STRIDE': [16],
 'GPU_ID': 0,
 'MATLAB': 'matlab',
 'MAX_NUM_GT_BOXES': 20,
 'MOBILENET': {'DEPTH_MULTIPLIER': 1.0,
               'FIXED_LAYERS': 5,
               'REGU_DEPTH': False,
               'WEIGHT_DECAY': 4e-05},
 'PIXEL_MEANS': array([[[102.9801, 115.9465, 122.7717]]]),
 'POOLING_MODE': 'align',
 'POOLING_SIZE': 7,
 'RESNET': {'FIXED_BLOCKS': 1, 'MAX_POOL': False},
 'RNG_SEED': 3,
 'ROOT_DIR': '/media/latest_mount/Dev/git/jwyang/faster-rcnn.pytorch',
 'TEST': {'BBOX_REG': True,
          'HAS_RPN': True,
          'MAX_SIZE': 1000,
          'MODE': 'nms',
          'NMS': 0.3,
          'PROPOSAL_METHOD': 'gt',
          'RPN_MIN_SIZE': 16,
          'RPN_NMS_THRESH': 0.7,
          'RPN_POST_NMS_TOP_N': 300,
          'RPN_PRE_NMS_TOP_N': 6000,
          'RPN_TOP_N': 5000,
          'SCALES': [600],
          'SVM': False},
 'TRAIN': {'ASPECT_GROUPING': False,
           'BATCH_SIZE': 128,
           'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0],
           'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2],
           'BBOX_NORMALIZE_TARGETS': True,
           'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': True,
           'BBOX_REG': True,
           'BBOX_THRESH': 0.5,
           'BG_THRESH_HI': 0.5,
           'BG_THRESH_LO': 0.0,
           'BIAS_DECAY': False,
           'BN_TRAIN': False,
           'DISPLAY': 20,
           'DOUBLE_BIAS': False,
           'FG_FRACTION': 0.25,
           'FG_THRESH': 0.5,
           'GAMMA': 0.1,
           'HAS_RPN': True,
           'IMS_PER_BATCH': 1,
           'LEARNING_RATE': 0.001,
           'MAX_SIZE': 1000,
           'MOMENTUM': 0.9,
           'PROPOSAL_METHOD': 'gt',
           'RPN_BATCHSIZE': 256,
           'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'RPN_CLOBBER_POSITIVES': False,
           'RPN_FG_FRACTION': 0.5,
           'RPN_MIN_SIZE': 8,
           'RPN_NEGATIVE_OVERLAP': 0.3,
           'RPN_NMS_THRESH': 0.7,
           'RPN_POSITIVE_OVERLAP': 0.7,
           'RPN_POSITIVE_WEIGHT': -1.0,
           'RPN_POST_NMS_TOP_N': 2000,
           'RPN_PRE_NMS_TOP_N': 12000,
           'SCALES': [600],
           'SNAPSHOT_ITERS': 5000,
           'SNAPSHOT_KEPT': 3,
           'SNAPSHOT_PREFIX': 'res101_faster_rcnn',
           'STEPSIZE': [30000],
           'SUMMARY_INTERVAL': 180,
           'TRIM_HEIGHT': 600,
           'TRIM_WIDTH': 600,
           'TRUNCATED': False,
           'USE_ALL_GT': True,
           'USE_FLIPPED': True,
           'USE_GT': False,
           'WEIGHT_DECAY': 0.0001},
 'USE_GPU_NMS': True}
Loaded dataset `voc_2007_test` for training
Set proposal method: gt
Preparing training data...
wrote gt roidb to /media/latest_mount/Dev/git/jwyang/faster-rcnn.pytorch/data/cache/voc_2007_test_gt_roidb.pkl
done
4952 roidb entries
load checkpoint models/res101/pascal_voc/faster_rcnn_1_1_10021.pth
load model successfully!
Evaluating detections0.140s 0.010s   
Writing aeroplane VOC results file
Writing bicycle VOC results file
Writing bird VOC results file
Writing boat VOC results file
Writing bottle VOC results file
Writing bus VOC results file
Writing car VOC results file
Writing cat VOC results file
Writing chair VOC results file
Writing cow VOC results file
Writing diningtable VOC results file
Writing dog VOC results file
Writing horse VOC results file
Writing motorbike VOC results file
Writing person VOC results file
Writing pottedplant VOC results file
Writing sheep VOC results file
Writing sofa VOC results file
Writing train VOC results file
Writing tvmonitor VOC results file
VOC07 metric? Yes
Reading annotation for 1/4952
Reading annotation for 101/4952
Reading annotation for 201/4952
Reading annotation for 301/4952
Reading annotation for 401/4952
Reading annotation for 501/4952
Reading annotation for 601/4952
Reading annotation for 701/4952
Reading annotation for 801/4952
Reading annotation for 901/4952
Reading annotation for 1001/4952
Reading annotation for 1101/4952
Reading annotation for 1201/4952
Reading annotation for 1301/4952
Reading annotation for 1401/4952
Reading annotation for 1501/4952
Reading annotation for 1601/4952
Reading annotation for 1701/4952
Reading annotation for 1801/4952
Reading annotation for 1901/4952
Reading annotation for 2001/4952
Reading annotation for 2101/4952
Reading annotation for 2201/4952
Reading annotation for 2301/4952
Reading annotation for 2401/4952
Reading annotation for 2501/4952
Reading annotation for 2601/4952
Reading annotation for 2701/4952
Reading annotation for 2801/4952
Reading annotation for 2901/4952
Reading annotation for 3001/4952
Reading annotation for 3101/4952
Reading annotation for 3201/4952
Reading annotation for 3301/4952
Reading annotation for 3401/4952
Reading annotation for 3501/4952
Reading annotation for 3601/4952
Reading annotation for 3701/4952
Reading annotation for 3801/4952
Reading annotation for 3901/4952
Reading annotation for 4001/4952
Reading annotation for 4101/4952
Reading annotation for 4201/4952
Reading annotation for 4301/4952
Reading annotation for 4401/4952
Reading annotation for 4501/4952
Reading annotation for 4601/4952
Reading annotation for 4701/4952
Reading annotation for 4801/4952
Reading annotation for 4901/4952
Saving cached annotations to /media/latest_mount/Dev/git/jwyang/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt_annots.pkl
AP for aeroplane = 0.5231
AP for bicycle = 0.7219
AP for bird = 0.6763
AP for boat = 0.4601
AP for bottle = 0.4647
AP for bus = 0.5729
AP for car = 0.7456
AP for cat = 0.7314
AP for chair = 0.3267
AP for cow = 0.6898
AP for diningtable = 0.4868
AP for dog = 0.6852
AP for horse = 0.7820
AP for motorbike = 0.6141
AP for person = 0.6890
AP for pottedplant = 0.3821
AP for sheep = 0.6027
AP for sofa = 0.5595
AP for train = 0.5494
AP for tvmonitor = 0.5683
Mean AP = 0.5916
~~~~~~~~
Results:
0.523
0.722
0.676
0.460
0.465
0.573
0.746
0.731
0.327
0.690
0.487
0.685
0.782
0.614
0.689
0.382
0.603
0.559
0.549
0.568
0.592
~~~~~~~~

--------------------------------------------------------------
Results computed with the **unofficial** Python eval code.
Results should be very close to the official MATLAB eval code.
Recompute with `./tools/reval.py --matlab ...` for your paper.
-- Thanks, The Management
--------------------------------------------------------------
test time: 860.4468s

```

When I correctly configured the test to test the model I had rolled... 

```
$ python test_net.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 20 --cuda --load_dir models/

```
which spat out

```
Called with args:
Namespace(cfg_file='cfgs/vgg16.yml', checkepoch=20, checkpoint=10021, checksession=1, class_agnostic=False, cuda=True, dataset='pascal_voc', large_scale=False, load_dir='models/', mGPUs=False, net='res101', parallel_type=0, set_cfgs=None, vis=False)
Using config:
{'ANCHOR_RATIOS': [0.5, 1, 2],
 'ANCHOR_SCALES': [8, 16, 32],
 'CROP_RESIZE_WITH_MAX_POOL': False,
 'CUDA': False,
 'DATA_DIR': '/media/latest_mount/Dev/git/jwyang/faster-rcnn.pytorch/data',
 'DEDUP_BOXES': 0.0625,
 'EPS': 1e-14,
 'EXP_DIR': 'res101',
 'FEAT_STRIDE': [16],
 'GPU_ID': 0,
 'MATLAB': 'matlab',
 'MAX_NUM_GT_BOXES': 20,
 'MOBILENET': {'DEPTH_MULTIPLIER': 1.0,
               'FIXED_LAYERS': 5,
               'REGU_DEPTH': False,
               'WEIGHT_DECAY': 4e-05},
 'PIXEL_MEANS': array([[[102.9801, 115.9465, 122.7717]]]),
 'POOLING_MODE': 'align',
 'POOLING_SIZE': 7,
 'RESNET': {'FIXED_BLOCKS': 1, 'MAX_POOL': False},
 'RNG_SEED': 3,
 'ROOT_DIR': '/media/latest_mount/Dev/git/jwyang/faster-rcnn.pytorch',
 'TEST': {'BBOX_REG': True,
          'HAS_RPN': True,
          'MAX_SIZE': 1000,
          'MODE': 'nms',
          'NMS': 0.3,
          'PROPOSAL_METHOD': 'gt',
          'RPN_MIN_SIZE': 16,
          'RPN_NMS_THRESH': 0.7,
          'RPN_POST_NMS_TOP_N': 300,
          'RPN_PRE_NMS_TOP_N': 6000,
          'RPN_TOP_N': 5000,
          'SCALES': [600],
          'SVM': False},
 'TRAIN': {'ASPECT_GROUPING': False,
           'BATCH_SIZE': 128,
           'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0],
           'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2],
           'BBOX_NORMALIZE_TARGETS': True,
           'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': True,
           'BBOX_REG': True,
           'BBOX_THRESH': 0.5,
           'BG_THRESH_HI': 0.5,
           'BG_THRESH_LO': 0.0,
           'BIAS_DECAY': False,
           'BN_TRAIN': False,
           'DISPLAY': 20,
           'DOUBLE_BIAS': False,
           'FG_FRACTION': 0.25,
           'FG_THRESH': 0.5,
           'GAMMA': 0.1,
           'HAS_RPN': True,
           'IMS_PER_BATCH': 1,
           'LEARNING_RATE': 0.001,
           'MAX_SIZE': 1000,
           'MOMENTUM': 0.9,
           'PROPOSAL_METHOD': 'gt',
           'RPN_BATCHSIZE': 256,
           'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'RPN_CLOBBER_POSITIVES': False,
           'RPN_FG_FRACTION': 0.5,
           'RPN_MIN_SIZE': 8,
           'RPN_NEGATIVE_OVERLAP': 0.3,
           'RPN_NMS_THRESH': 0.7,
           'RPN_POSITIVE_OVERLAP': 0.7,
           'RPN_POSITIVE_WEIGHT': -1.0,
           'RPN_POST_NMS_TOP_N': 2000,
           'RPN_PRE_NMS_TOP_N': 12000,
           'SCALES': [600],
           'SNAPSHOT_ITERS': 5000,
           'SNAPSHOT_KEPT': 3,
           'SNAPSHOT_PREFIX': 'res101_faster_rcnn',
           'STEPSIZE': [30000],
           'SUMMARY_INTERVAL': 180,
           'TRIM_HEIGHT': 600,
           'TRIM_WIDTH': 600,
           'TRUNCATED': False,
           'USE_ALL_GT': True,
           'USE_FLIPPED': True,
           'USE_GT': False,
           'WEIGHT_DECAY': 0.0001},
 'USE_GPU_NMS': True}
Loaded dataset `voc_2007_test` for training
Set proposal method: gt
Preparing training data...
voc_2007_test gt roidb loaded from /media/latest_mount/Dev/git/jwyang/faster-rcnn.pytorch/data/cache/voc_2007_test_gt_roidb.pkl
done
4952 roidb entries
load checkpoint models//res101/pascal_voc/faster_rcnn_1_20_10021.pth
load model successfully!
Evaluating detections0.139s 0.010s   
Writing aeroplane VOC results file
Writing bicycle VOC results file
Writing bird VOC results file
Writing boat VOC results file
Writing bottle VOC results file
Writing bus VOC results file
Writing car VOC results file
Writing cat VOC results file
Writing chair VOC results file
Writing cow VOC results file
Writing diningtable VOC results file
Writing dog VOC results file
Writing horse VOC results file
Writing motorbike VOC results file
Writing person VOC results file
Writing pottedplant VOC results file
Writing sheep VOC results file
Writing sofa VOC results file
Writing train VOC results file
Writing tvmonitor VOC results file
VOC07 metric? Yes
AP for aeroplane = 0.7512
AP for bicycle = 0.8028
AP for bird = 0.7430
AP for boat = 0.6511
AP for bottle = 0.5912
AP for bus = 0.7758
AP for car = 0.8216
AP for cat = 0.8682
AP for chair = 0.5382
AP for cow = 0.8152
AP for diningtable = 0.6646
AP for dog = 0.8455
AP for horse = 0.8509
AP for motorbike = 0.7762
AP for person = 0.7808
AP for pottedplant = 0.4533
AP for sheep = 0.7524
AP for sofa = 0.7150
AP for train = 0.7716
AP for tvmonitor = 0.7217
Mean AP = 0.7345
~~~~~~~~
Results:
0.751
0.803
0.743
0.651
0.591
0.776
0.822
0.868
0.538
0.815
0.665
0.845
0.851
0.776
0.781
0.453
0.752
0.715
0.772
0.722
0.735
~~~~~~~~

--------------------------------------------------------------
Results computed with the **unofficial** Python eval code.
Results should be very close to the official MATLAB eval code.
Recompute with `./tools/reval.py --matlab ...` for your paper.
-- Thanks, The Management
--------------------------------------------------------------
test time: 862.8947s
```

a subsequent run repeated the results.. which is a good sign


Running model again now
```
(pytorch1_py36) emcp@emcp-corei7-2600k:/media/latest_mount/Dev/git/EMCP/faster-rcnn.pytorch$ CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net res101 --bs 1 --nw 1 --lr .001 --lr_decay_step 10 --cuda --epochs 21
Called with args:
Namespace(batch_size=1, checkepoch=1, checkpoint=0, checkpoint_interval=10000, checksession=1, class_agnostic=False, cuda=True, dataset='pascal_voc', disp_interval=100, large_scale=False, lr=0.001, lr_decay_gamma=0.1, lr_decay_step=10, mGPUs=False, max_epochs=21, net='res101', num_workers=1, optimizer='sgd', resume=False, save_dir='models', session=1, start_epoch=1, use_tfboard=False)
Using config:
{'ANCHOR_RATIOS': [0.5, 1, 2],
 'ANCHOR_SCALES': [8, 16, 32],
 'CROP_RESIZE_WITH_MAX_POOL': False,
 'CUDA': False,
 'DATA_DIR': '/media/latest_mount/Dev/git/EMCP/faster-rcnn.pytorch/data',
 'DEDUP_BOXES': 0.0625,
 'EPS': 1e-14,
 'EXP_DIR': 'res101',
 'FEAT_STRIDE': [16],
 'GPU_ID': 0,
 'MATLAB': 'matlab',
 'MAX_NUM_GT_BOXES': 20,
 'MOBILENET': {'DEPTH_MULTIPLIER': 1.0,
               'FIXED_LAYERS': 5,
               'REGU_DEPTH': False,
               'WEIGHT_DECAY': 4e-05},
 'PIXEL_MEANS': array([[[102.9801, 115.9465, 122.7717]]]),
 'POOLING_MODE': 'align',
 'POOLING_SIZE': 7,
 'RESNET': {'FIXED_BLOCKS': 1, 'MAX_POOL': False},
 'RNG_SEED': 3,
 'ROOT_DIR': '/media/latest_mount/Dev/git/EMCP/faster-rcnn.pytorch',
 'TEST': {'BBOX_REG': True,
          'HAS_RPN': True,
          'MAX_SIZE': 1000,
          'MODE': 'nms',
          'NMS': 0.3,
          'PROPOSAL_METHOD': 'gt',
          'RPN_MIN_SIZE': 16,
          'RPN_NMS_THRESH': 0.7,
          'RPN_POST_NMS_TOP_N': 300,
          'RPN_PRE_NMS_TOP_N': 6000,
          'RPN_TOP_N': 5000,
          'SCALES': [600],
          'SVM': False},
 'TRAIN': {'ASPECT_GROUPING': False,
           'BATCH_SIZE': 128,
           'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0],
           'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2],
           'BBOX_NORMALIZE_TARGETS': True,
           'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': True,
           'BBOX_REG': True,
           'BBOX_THRESH': 0.5,
           'BG_THRESH_HI': 0.5,
           'BG_THRESH_LO': 0.0,
           'BIAS_DECAY': False,
           'BN_TRAIN': False,
           'DISPLAY': 20,
           'DOUBLE_BIAS': False,
           'FG_FRACTION': 0.25,
           'FG_THRESH': 0.5,
           'GAMMA': 0.1,
           'HAS_RPN': True,
           'IMS_PER_BATCH': 1,
           'LEARNING_RATE': 0.001,
           'MAX_SIZE': 1000,
           'MOMENTUM': 0.9,
           'PROPOSAL_METHOD': 'gt',
           'RPN_BATCHSIZE': 256,
           'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'RPN_CLOBBER_POSITIVES': False,
           'RPN_FG_FRACTION': 0.5,
           'RPN_MIN_SIZE': 8,
           'RPN_NEGATIVE_OVERLAP': 0.3,
           'RPN_NMS_THRESH': 0.7,
           'RPN_POSITIVE_OVERLAP': 0.7,
           'RPN_POSITIVE_WEIGHT': -1.0,
           'RPN_POST_NMS_TOP_N': 2000,
           'RPN_PRE_NMS_TOP_N': 12000,
           'SCALES': [600],
           'SNAPSHOT_ITERS': 5000,
           'SNAPSHOT_KEPT': 3,
           'SNAPSHOT_PREFIX': 'res101_faster_rcnn',
           'STEPSIZE': [30000],
           'SUMMARY_INTERVAL': 180,
           'TRIM_HEIGHT': 600,
           'TRIM_WIDTH': 600,
           'TRUNCATED': False,
           'USE_ALL_GT': True,
           'USE_FLIPPED': True,
           'USE_GT': False,
           'WEIGHT_DECAY': 0.0001},
 'USE_GPU_NMS': True}
Loaded dataset `voc_2007_trainval` for training
Set proposal method: gt
Appending horizontally-flipped training examples...
wrote gt roidb to /media/latest_mount/Dev/git/EMCP/faster-rcnn.pytorch/data/cache/voc_2007_trainval_gt_roidb.pkl
done
Preparing training data...
done
before filtering, there are 10022 images...
after filtering, there are 10022 images...
10022 roidb entries
Loading pretrained weights from data/pretrained_model/resnet101_caffe.pth
[session 1][epoch  1][iter    0/10022] loss: 4.6673, lr: 1.00e-03
			fg/bg=(32/96), time cost: 0.485099
			rpn_cls: 0.6445, rpn_box: 0.2561, rcnn_cls: 3.1042, rcnn_box 0.6625

```

21 total epochs took 19 hours

```
[session 1][epoch 21][iter 9800/10022] loss: 0.1174, lr: 1.00e-04
			fg/bg=(23/105), time cost: 32.350767
			rpn_cls: 0.0000, rpn_box: 0.0002, rcnn_cls: 0.0047, rcnn_box 0.0063
[session 1][epoch 21][iter 9900/10022] loss: 0.1053, lr: 1.00e-04
			fg/bg=(32/96), time cost: 32.574554
			rpn_cls: 0.0086, rpn_box: 0.0385, rcnn_cls: 0.0510, rcnn_box 0.0956
[session 1][epoch 21][iter 10000/10022] loss: 0.1097, lr: 1.00e-04
			fg/bg=(32/96), time cost: 32.532319
			rpn_cls: 0.0041, rpn_box: 0.0460, rcnn_cls: 0.0969, rcnn_box 0.1026

save model: models/res101/pascal_voc/faster_rcnn_1_21_10021.pth
START Time 1551908591.1611655
END Time 1551977412.2082477
TOTAL RUNTIME WAS 68821.04708218575 seconds .. or 
d:h:m:s -> 0:19:7:1
(pytorch1_py36) emcp@emcp-corei7-2600k:/media/latest_mount/Dev/git/EMCP/faster-rcnn.pytorch$ CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net res101 --bs 1 --nw 1 --lr .001 --lr_decay_step 10 --cuda --epochs 40
Called with args:

```

scaling up to 40 epochs now

```
(pytorch1_py36) emcp@emcp-corei7-2600k:/media/latest_mount/Dev/git/EMCP/faster-rcnn.pytorch$ CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net res101 --bs 1 --nw 1 --lr .001 --lr_decay_step 10 --cuda --epochs 40
Called with args:
Namespace(batch_size=1, checkepoch=1, checkpoint=0, checkpoint_interval=10000, checksession=1, class_agnostic=False, cuda=True, dataset='pascal_voc', disp_interval=100, large_scale=False, lr=0.001, lr_decay_gamma=0.1, lr_decay_step=10, mGPUs=False, max_epochs=40, net='res101', num_workers=1, optimizer='sgd', resume=False, save_dir='models', session=1, start_epoch=1, use_tfboard=False)
Using config:
{'ANCHOR_RATIOS': [0.5, 1, 2],
 'ANCHOR_SCALES': [8, 16, 32],
 'CROP_RESIZE_WITH_MAX_POOL': False,
 'CUDA': False,
 'DATA_DIR': '/media/latest_mount/Dev/git/EMCP/faster-rcnn.pytorch/data',
 'DEDUP_BOXES': 0.0625,
 'EPS': 1e-14,
 'EXP_DIR': 'res101',
 'FEAT_STRIDE': [16],
 'GPU_ID': 0,
 'MATLAB': 'matlab',
 'MAX_NUM_GT_BOXES': 20,
 'MOBILENET': {'DEPTH_MULTIPLIER': 1.0,
               'FIXED_LAYERS': 5,
               'REGU_DEPTH': False,
               'WEIGHT_DECAY': 4e-05},
 'PIXEL_MEANS': array([[[102.9801, 115.9465, 122.7717]]]),
 'POOLING_MODE': 'align',
 'POOLING_SIZE': 7,
 'RESNET': {'FIXED_BLOCKS': 1, 'MAX_POOL': False},
 'RNG_SEED': 3,
 'ROOT_DIR': '/media/latest_mount/Dev/git/EMCP/faster-rcnn.pytorch',
 'TEST': {'BBOX_REG': True,
          'HAS_RPN': True,
          'MAX_SIZE': 1000,
          'MODE': 'nms',
          'NMS': 0.3,
          'PROPOSAL_METHOD': 'gt',
          'RPN_MIN_SIZE': 16,
          'RPN_NMS_THRESH': 0.7,
          'RPN_POST_NMS_TOP_N': 300,
          'RPN_PRE_NMS_TOP_N': 6000,
          'RPN_TOP_N': 5000,
          'SCALES': [600],
          'SVM': False},
 'TRAIN': {'ASPECT_GROUPING': False,
           'BATCH_SIZE': 128,
           'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0],
           'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2],
           'BBOX_NORMALIZE_TARGETS': True,
           'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': True,
           'BBOX_REG': True,
           'BBOX_THRESH': 0.5,
           'BG_THRESH_HI': 0.5,
           'BG_THRESH_LO': 0.0,
           'BIAS_DECAY': False,
           'BN_TRAIN': False,
           'DISPLAY': 20,
           'DOUBLE_BIAS': False,
           'FG_FRACTION': 0.25,
           'FG_THRESH': 0.5,
           'GAMMA': 0.1,
           'HAS_RPN': True,
           'IMS_PER_BATCH': 1,
           'LEARNING_RATE': 0.001,
           'MAX_SIZE': 1000,
           'MOMENTUM': 0.9,
           'PROPOSAL_METHOD': 'gt',
           'RPN_BATCHSIZE': 256,
           'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'RPN_CLOBBER_POSITIVES': False,
           'RPN_FG_FRACTION': 0.5,
           'RPN_MIN_SIZE': 8,
           'RPN_NEGATIVE_OVERLAP': 0.3,
           'RPN_NMS_THRESH': 0.7,
           'RPN_POSITIVE_OVERLAP': 0.7,
           'RPN_POSITIVE_WEIGHT': -1.0,
           'RPN_POST_NMS_TOP_N': 2000,
           'RPN_PRE_NMS_TOP_N': 12000,
           'SCALES': [600],
           'SNAPSHOT_ITERS': 5000,
           'SNAPSHOT_KEPT': 3,
           'SNAPSHOT_PREFIX': 'res101_faster_rcnn',
           'STEPSIZE': [30000],
           'SUMMARY_INTERVAL': 180,
           'TRIM_HEIGHT': 600,
           'TRIM_WIDTH': 600,
           'TRUNCATED': False,
           'USE_ALL_GT': True,
           'USE_FLIPPED': True,
           'USE_GT': False,
           'WEIGHT_DECAY': 0.0001},
 'USE_GPU_NMS': True}
Loaded dataset `voc_2007_trainval` for training
Set proposal method: gt
Appending horizontally-flipped training examples...
voc_2007_trainval gt roidb loaded from /media/latest_mount/Dev/git/EMCP/faster-rcnn.pytorch/data/cache/voc_2007_trainval_gt_roidb.pkl
done
Preparing training data...
done
before filtering, there are 10022 images...
after filtering, there are 10022 images...
10022 roidb entries
Loading pretrained weights from data/pretrained_model/resnet101_caffe.pth
[session 1][epoch  1][iter    0/10022] loss: 4.3784, lr: 1.00e-03
			fg/bg=(29/99), time cost: 0.492602
			rpn_cls: 0.7079, rpn_box: 0.1296, rcnn_cls: 2.9030, rcnn_box 0.6379
[session 1][epoch  1][iter  100/10022] loss: 1.7180, lr: 1.00e-03
			fg/bg=(17/111), time cost: 32.317783
			rpn_cls: 0.0992, rpn_box: 0.0214, rcnn_cls: 0.6073, rcnn_box 0.3259
[session 1][epoch  1][iter  200/10022] loss: 1.4718, lr: 1.00e-03
			fg/bg=(23/105), time cost: 32.949185

```