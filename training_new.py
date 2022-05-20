import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from datetime import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
import cv2
import csv
import random
import gc
import nibabel as nib
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
from faster_rcnn_util import *
import configparser


# Faster RCNN module
sys.path.insert(0, '/tf/jacky831006/faster-rcnn.pytorch-0.4/lib/')
from model.utils.config_3d import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils_3d import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.faster_rcnn.resnet_3d import resnet
from model.rpn.bbox_transform_3d import bbox_transform_inv, clip_boxes

# Data augmnetation module (based on MONAI)

from monai.apps import download_and_extract
from monai.data import CacheDataset, DataLoader, Dataset
from monai.config import print_config
from monai.utils import first, set_determinism
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadNifti,
    LoadNiftid,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    CenterSpatialCropd,
    RandCropByPosNegLabeld,
    Resized,
    ToTensord
)
import functools
# let all of print can be flush = ture
print = functools.partial(print, flush=True)

# image mask to label (x1,y1,x2,y2,z1,z2)
'''
訓練的時候 不能使用CropForegroundd(把沒有影像的部份去掉)
而新的方法有label可以比對diou 因此要先用有沒CropForegroundd的方法再用有CropForegroundd的方法predict
'''
# Data hyperparameter import
# *_new means have new data 

train_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_train_20220310.csv')
valid_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_valid_20220310.csv')
test_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_test_20220310.csv')

cfgpath ='/tf/jacky831006/faster-rcnn.pytorch-0.4/config/standard_config_new_data_onelabel_9.ini'
conf = configparser.ConfigParser()
conf.read(cfgpath)

# Augmentation
num_samples = conf.getint('Augmentation','num_sample')
size = eval(conf.get('Augmentation','size'))
prob = conf.getfloat('RandAffined','prob')
translate_range = eval(conf.get('RandAffined','translate_range'))
rotate_range = eval(conf.get('RandAffined','rotate_range'))
scale_range = eval(conf.get('RandAffined','scale_range'))

# Data_setting
pre_train = conf.getboolean('Data_Setting','pretrained')
gpu_number = conf.get('Data_Setting','gpu')
seed = conf.getint('Data_Setting','seed')
cross_kfold = conf.getint('Data_Setting','cross_kfold')
epoch = conf.getint('Data_Setting','epoch')
early_stop = conf.getint('Data_Setting','early_stop')
traning_batch_size = conf.getint('Data_Setting','traning_batch_size')
valid_batch_size = conf.getint('Data_Setting','valid_batch_size')
testing_batch_size = conf.getint('Data_Setting','testing_batch_size')
data_split_ratio = eval(conf.get('Data_Setting','data_split_ratio'))
dataloader_num_workers = conf.getint('Data_Setting','dataloader_num_workers')
mGPU = conf.getboolean('Data_Setting','mGPU')
init_lr = conf.getfloat('Data_Setting','init_lr')
optimizer = conf.get('Data_Setting','optimizer')
lr_decay_rate = conf.getfloat('Data_Setting','lr_decay_rate')
lr_decay_epoch = conf.getint('Data_Setting','lr_decay_epoch')

# ------- data augmentation -------

# data augmentation

train_transforms = Compose(
    [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        # CropForeground 要在 resize前面
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=['image', 'label'], spatial_size = size),
        # windowing 要在 resize後面
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        #CenterSpatialCropd(keys=['image','label'],roi_size =(96,96,96)),
        Dulicated(keys= ["image","label"], num_samples = num_samples),
        # user can also add other random transforms
        RandAffined(keys=["image", "label"], 
            mode=("bilinear", "nearest"), 
            prob=prob,
            spatial_size=size,
            translate_range=translate_range,
            rotate_range=rotate_range, 
            scale_range=scale_range,
            padding_mode="border"),
        Annotate(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
        ])
        
'''
RandAffined(keys=["image", "label"], 
    mode=("bilinear", "nearest"), 
    prob=0.5,
    spatial_size=(128, 128, 128),
    translate_range=(20, 20, 5),
    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 15), # 180/36=5  180/15=16
    scale_range=(0.1, 0.1, 0.1),
    padding_mode="border"),

RandCropByPosNegLabeld(
    keys=["image", "label"],
    label_key="label",
    spatial_size=(128, 128, 128),
    pos=1,
    #neg=1,
    num_samples=4,
    image_key="image",
    image_threshold=0,
)

Rand3DElasticd(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=0.5,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        spatial_size=(128, 128, 128),
        translate_range=(20, 20, 5),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 15),
        scale_range=(0.1, 0.1, 0.1),
        padding_mode="border"
    ),
'''
val_transforms = Compose(
    [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Resized(keys=['image', 'label'], spatial_size = size),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        Annotate(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
    ]
)
'''
def adjust_learning_rate(optimizer, epoch, init_lr, decay_rate=.5 ,lr_decay_epoch=40):
    # Sets the learning rate to initial LR decayed by e^(-0.1*epochs)
    lr = init_lr * (decay_rate ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        #param_group['lr'] =  param_group['lr'] * math.exp(-decay_rate*epoch)
        param_group['lr'] = lr
        #lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    #print('LR is set to {}'.format(param_group['lr']))
    return optimizer , lr
'''

# start a typical PyTorch training
def train(model, epochs, optimizer, train_loader, valid_loader, early_stop, class_list):
    # Let ini config file can be writted
    global best_metric
    global best_metric_epoch
    best_metric = -1
    best_metric_epoch = -1
    trigger_times = 0

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        # why? need model.train().to(device)
        # setting to train mode
        #model.train().to(device)
        model.train()
        start = time.time()
        epoch_loss = 0
        step = 0
        # LR decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_rate, patience=lr_decay_epoch, verbose =True) 
        #optimizer , LR = adjust_learning_rate(optimizer, epoch, init_lr, decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch) 
        for batch_data in train_loader:
            step += 1
            
            im_data, gt_boxes, im_info, num_boxes  = batch_data['image'].cuda(), batch_data['label'].cuda(), \
                                                    batch_data['im_info'], batch_data['num_box'].cuda()
            
            #print(im_data.shape)
            model.zero_grad()
            
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = model(im_data, im_info, gt_boxes, num_boxes)
            
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            epoch_loss += loss.item()
            
            # backward
            # Zero the gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time.time()
            
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            #writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            if mGPUs:
                loss_rpn_cls = rpn_loss_cls.mean().item()
                loss_rpn_box = rpn_loss_box.mean().item()
                loss_rcnn_cls = RCNN_loss_cls.mean().item()
                loss_rcnn_box = RCNN_loss_bbox.mean().item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt
            else:
                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt
                
           # print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
           #                     % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
            print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
            print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        #print(f'LR is set to {LR}')
        
        # tfboard setting
        info = {
            'loss': epoch_loss,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
        }
        writer.add_scalars("logs_s_{}/losses".format(1), info, (epoch - 1) * epoch_len + step)
        epoch_loss = 0
        start = time.time()

        # Early stopping & save best weights by using validation
        metric = validation(model, valid_loader, class_list)
        scheduler.step(metric)

        # checkpoint setting
        if metric >= best_metric:
            # reset trigger_times
            trigger_times = 0
            best_metric = metric
            best_metric_epoch = epoch + 1
            
            torch.save(model.state_dict(), f"{check_path}/{best_metric}.pth")
            print('trigger times:', trigger_times)
            print("saved new best metric model")
        else:
            trigger_times += 1
            print('trigger times:', trigger_times)
            # Save last 3 epoch weight
            if early_stop - trigger_times <= 3:
                torch.save(model.state_dict(), f"{check_path}/{metric}_last.pth")
                print("save last metric model")
        print(
            "current epoch: {} current IOU: {:.4f} best IOU: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )
        writer.add_scalar("val_accuracy", metric, epoch + 1)

        if trigger_times >= early_stop:
            print('Early stopping!\nStart to test process.')
            print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
            return model
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

    return model


# batch size為 1的測試
def validation(model, val_loader, class_list):
    # Settings
    model.eval()
    # all_boxes[0] 為背景
    # all_boxes[1] 為spleen，而all_boxes[1][0]~all_boxes[1][3] 不同batch的boxes 結果，可能會有多個
    all_boxes = [[[] for _ in range(len(valid_data_dicts))]
                for _ in range(len(class_list))]
    all_iou = [[[] for _ in range(len(valid_data_dicts))]
                for _ in range(len(class_list))]
    
    # Test validation data
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        i = -1
        empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
        for val_data in val_loader:
            # val_loader的順序
            i += 1
            # val_images, val_labels = val_data['image'].to(device), val_data['class'].to(device)
            im_data, gt_boxes, im_info, num_boxes  = val_data['image'].cuda(), val_data['label'].cuda(), \
                                                     val_data['im_info'], val_data['num_box'].cuda()
            ori_shape = val_data['image_meta_dict']['spatial_shape'].cuda()
            det_tic = time.time()
            #val_outputs = model(val_images)
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = model(im_data, im_info, gt_boxes, num_boxes)
            
            scores = cls_prob.data
            boxes = rois.data[:, :, 1:7]
            #print(f'Valid predict box:{boxes}')
            #print(f'Valid predict score:{scores}')
            if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    #if args.class_agnostic:
                    #    box_deltas = box_deltas.view(-1, 6) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    #               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    #    box_deltas = box_deltas.view(1, -1, 6)
                    #else:
                    box_deltas = box_deltas.view(-1, 6) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(val_loader.batch_size, -1, 6 * len(class_list))
                #print(f'Valid box_deltas:{box_deltas.shape}')
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
                pred_boxes = _.cuda() # if args.cuda > 0 else _
            
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            # Caculate gt box and box IOU
            gt_boxes = gt_boxes.squeeze(0)
            det_toc = time.time()
            detect_time = det_toc - det_tic
            #print(f'detect_time:{detect_time}')
            misc_tic = time.time()
            # check output is same as label if same return ture else false
            #if vis:
            #    im = cv2.imread(imdb.image_path_at(i))
            #    im2show = np.copy(im)
            #print(f'pred_boxes last:{pred_boxes}')
            for j in range(1, len(class_list)):
                inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                # if there is det
                if inds.numel() > 0:  
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    #if args.class_agnostic:
                    #    cls_boxes = pred_boxes[inds, :]
                    #else:
                    # 0:6 是 bg
                    cls_boxes = pred_boxes[inds][:, j * 6:(j + 1) * 6]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    #keep = nms(cls_dets, cfg.TEST.NMS)
                    #print(f'cls_score:{cls_dets[:,-1].shape}')
                    anchorBoxes_f, keep, scores_f = Weighted_cluster_nms(cls_dets[:,:6], cls_dets[:,-1].view(-1,1), 0.3)
                    cls_dets_last = torch.cat((anchorBoxes_f, scores_f), 1)
                    #print(f'cls_dets_last shape:{cls_dets_last.shape}')
                    
                    # gt_boxes : (batch_size, num, 7)
                    # Caculate GT box and Predict box IOU (Only one gt_box)
                    iou = calc_diou(gt_boxes[:,:6].float(),cls_dets_last[:,:6])
                    # TP (1,num)
                    # AP 計算的方法在只有一個gt box時相對沒有意義
                    #TP = iou.ge(0.5).cpu().numpy()
                    #FP = iou.le(0.5).cpu().numpy() 
                    #Precision = TP.sum()/ TP.sum() + FP.sum()
                    #print(f"TP,FP:{(TP.shape,FP.shape)}")
                    
                    all_iou[j][i] = iou.cpu().numpy()
                    
                    #for k in range(cls_dets_last.size(0)):
                    #    calc_iou(gt_boxes,cls_dets_last[:,])
                    
                    #if vis:
                    #    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                
                    #print(f'cls_dets:{cls_dets}')
        
                    all_boxes[j][i] = cls_dets_last.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array
                    all_iou[j][i] = 0
            
            misc_toc = time.time()
            nms_time = misc_toc - misc_tic
            #print(f'nms_time:{nms_time}')

    # Caculate IOU mean as valid select condition
    #final_iou=np.array([i.mean() for i in all_iou[1]]).mean()
    final_iou = np.array([sum(i[0])/len(i[0]) for i in all_iou[1]]).mean()

    #df = pd.DataFrame(columns = ["source", "file","label","pixel_dimention"])
    #df_series = pd.Series([source_url, file_url, label_url], index=df.columns)    
        
    return final_iou


# batch size為 1的測試
def test(model, test_loader, class_list):
    # Settings
    model.eval()
    # all_boxes[0] 為背景
    # all_boxes[1] 為spleen，而all_boxes[1][0]~all_boxes[1][3] 不同batch的boxes 結果，可能會有多個
    
    all_boxes = [[[] for _ in range(len(test_data_dicts))]
                for _ in range(len(class_list))]
    all_iou = [[[] for _ in range(len(test_data_dicts))]
                for _ in range(len(class_list))]
    
    # Test validation data
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        i = -1
        empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
        for test_data in test_loader:
            # val_loader的順序
            i += 1
            # val_images, val_labels = val_data['image'].to(device), val_data['class'].to(device)
            im_data, gt_boxes, im_info, num_boxes  = test_data['image'].cuda(), test_data['label'].cuda(), \
                                                     test_data['im_info'], test_data['num_box'].cuda()
            ori_shape = test_data['image_meta_dict']['spatial_shape'].cuda()
            det_tic = time.time()
            #val_outputs = model(val_images)
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = model(im_data, im_info, gt_boxes, num_boxes)
            
            scores = cls_prob.data
            boxes = rois.data[:, :, 1:7]
            #print(f'Valid predict box:{boxes}')
            #print(f'Valid predict score:{scores}')
            if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    #if args.class_agnostic:
                    #    box_deltas = box_deltas.view(-1, 6) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    #               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    #    box_deltas = box_deltas.view(1, -1, 6)
                    #else:
                    box_deltas = box_deltas.view(-1, 6) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(test_loader.batch_size, -1, 6 * len(class_list))
                #print(f'Valid box_deltas:{box_deltas.shape}')
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
                pred_boxes = _.cuda() # if args.cuda > 0 else _
            
            #print(f'Pred_boxes:{pred_boxes.shape}')
            #print(f'Scores:{scores}')
            #print(f'data[1][0][2] item:{data[1][0][2].item()}')
            
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            # Caculate gt box and box IOU
            gt_boxes = gt_boxes.squeeze(0)
            det_toc = time.time()
            detect_time = det_toc - det_tic
            #print(f'detect_time:{detect_time}')
            misc_tic = time.time()
            # check output is same as label if same return ture else false
            #if vis:
            #    im = cv2.imread(imdb.image_path_at(i))
            #    im2show = np.copy(im)
            #print(f'pred_boxes last:{pred_boxes}')
            for j in range(1, len(class_list)):
                inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                # if there is det
                if inds.numel() > 0:  
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    #if args.class_agnostic:
                    #    cls_boxes = pred_boxes[inds, :]
                    #else:
                    # 0:6 是 bg
                    cls_boxes = pred_boxes[inds][:, j * 6:(j + 1) * 6]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    #keep = nms(cls_dets, cfg.TEST.NMS)
                    #print(f'cls_score:{cls_dets[:,-1].shape}')
                    anchorBoxes_f, keep, scores_f = Weighted_cluster_nms(cls_dets[:,:6], cls_dets[:,-1].view(-1,1), 0.3)
                    cls_dets_last = torch.cat((anchorBoxes_f, scores_f), 1)
                    #print(f'cls_dets_last shape:{cls_dets_last.shape}')
                    
                    # gt_boxes : (batch_size, num, 7)
                    # Caculate GT box and Predict box IOU (Only one gt_box)
                    iou = calc_iou(gt_boxes[:,:6].float(),cls_dets_last[:,:6])
                    # TP (1,num)
                    # AP 計算的方法在只有一個gt box時相對沒有意義
                    #TP = iou.ge(0.5).cpu().numpy()
                    #FP = iou.le(0.5).cpu().numpy()
                    #Precision = TP.sum()/ TP.sum() + FP.sum()
                    #print(f"TP,FP:{(TP.shape,FP.shape)}")
                    
                    all_iou[j][i] = iou.cpu().numpy()
                    
                    #for k in range(cls_dets_last.size(0)):
                    #    calc_iou(gt_boxes,cls_dets_last[:,])
                    
                    #if vis:
                    #    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                
                    #print(f'cls_dets:{cls_dets}')
        
                    all_boxes[j][i] = cls_dets_last.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array
            
            
            misc_toc = time.time()
            nms_time = misc_toc - misc_tic
            #print(f'nms_time:{nms_time}')

    # Caculate IOU mean as valid select condition
    final_iou=np.array([i.mean() for i in all_iou[1]]).mean()


    
    #df = pd.DataFrame(columns = ["source", "file","label","pixel_dimention"])
    #df_series = pd.Series([source_url, file_url, label_url], index=df.columns)    
        
    return final_iou

#---------------- Data import ----------------
'''
with open("/tf/jacky831006/faster-rcnn.pytorch-0.4/ob_train_list.csv","r") as csvfile:
    rows = csv.DictReader(csvfile)
    data_dic = []
    for row in rows:
        data_dic.append({'image':row['image'],'label':row['label']})
'''

'''
# Old data
# one label only spleen
# class list
all_cls = ['__background__','spleen']
all_cls_label = [i for i in range(len(all_cls))]
all_cls_dic = dict(zip(all_cls,all_cls_label))

list_ =  ['pos','neg']
data = {}
for j in list_:
    image_path = f'/tf/jacky831006/object_detect_data_new/{j}/image'
    label_path = f'/tf/jacky831006/object_detect_data_new/{j}/label'
    label_list = os.listdir(label_path)

    data[j] = []
    for i in label_list:
        image = f'{image_path}/{i}'
        label = f'{label_path}/{i}'
        cls = 'spleen'
        all_cls = all_cls_dic
        
        data[j].append({'image':image,'label':label,'class':cls,'all_class':all_cls})

if cross_kfold*data_split_ratio[2] != 1 and cross_kfold!=1:
    raise RuntimeError("Kfold number is not match test data ratio")

first_start_time = time.time()
if cross_kfold==1:
    kf = KFold(n_splits=int(1/data_split_ratio[2]),shuffle=True,random_state=seed)
else:
    kf = KFold(n_splits=cross_kfold,shuffle=True,random_state=seed)

data_pos_index = dir()
data_num = 0
for train_index, test_index in kf.split(data['pos']):
    data_pos_index[data_num] = [train_index,test_index]
    data_num += 1

data_neg_index = dir()
data_num = 0
for train_index, test_index in kf.split(data['neg']):
    data_neg_index[data_num] = [train_index,test_index]
    data_num += 1

accuracy_list = []
test_diou_list = []
file_list = []
epoch_list = []

for k in range(cross_kfold):
    train_data_dicts_pos, valid_data_dicts_pos = train_test_split([data['pos'][i] for i in data_pos_index[k][0]], test_size=data_split_ratio[1]/(1-data_split_ratio[2]) ,random_state=seed)
    test_data_dicts_pos = [data['pos'][i] for i in data_pos_index[k][1]]

    train_data_dicts_neg, valid_data_dicts_neg = train_test_split([data['neg'][i] for i in data_neg_index[k][0]], test_size=data_split_ratio[1]/(1-data_split_ratio[2]) ,random_state=seed)
    test_data_dicts_neg = [data['neg'][i] for i in data_neg_index[k][1]]
    
    train_data_dicts = train_data_dicts_pos + train_data_dicts_neg
    valid_data_dicts = valid_data_dicts_pos + valid_data_dicts_neg
    test_data_dicts = test_data_dicts_pos + test_data_dicts_neg

    print(f'\n Train:{len(train_data_dicts)},Valid:{len(valid_data_dicts)},Test:{len(test_data_dicts)}')
    # set augmentation seed 
    set_determinism(seed=0)
    train_ds = CacheDataset(data=train_data_dicts, transform=train_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    train_data = DataLoader(train_ds, batch_size=traning_batch_size, shuffle=True, num_workers=dataloader_num_workers)

    valid_ds = CacheDataset(data=valid_data_dicts, transform=val_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    valid_data = DataLoader(valid_ds, batch_size=valid_batch_size, num_workers=dataloader_num_workers)

    test_ds = CacheDataset(data=test_data_dicts, transform=val_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    test_data = DataLoader(test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers)
'''
all_cls = ['__background__','spleen']
all_cls_label = [i for i in range(len(all_cls))]
all_cls_dic = dict(zip(all_cls,all_cls_label))


accuracy_list = []
test_diou_list = []
file_list = []
epoch_list = []

first_start_time = time.time()
def data_progress(df, dicts):
    for index, row in df.iterrows():
    
        if row['spleen_injury'] == 0:
            image = row['source']
            label = row['source'].replace('image','label')
        else:
            image = f'/tf/jacky831006/object_detect_data_new/pos/image/{row["chartNo"]}@venous_phase.nii.gz'
            label = f'/tf/jacky831006/object_detect_data_new/pos/label/{row["chartNo"]}@venous_phase.nii.gz'
        cls = 'spleen'
        all_cls = all_cls_dic
        dicts.append({'image':image,'label':label,'class':cls,'all_class':all_cls})
    return dicts

train_data_dicts = []
valid_data_dicts = []
test_data_dicts = []
train_data_dicts = data_progress(train_df,train_data_dicts)
valid_data_dicts = data_progress(valid_df,valid_data_dicts)
test_data_dicts = data_progress(test_df,test_data_dicts)

print(f'\n Train:{len(train_data_dicts)},Valid:{len(valid_data_dicts)},Test:{len(test_data_dicts)}')
# set augmentation seed 
set_determinism(seed=0)
train_ds = CacheDataset(data=train_data_dicts, transform=train_transforms, cache_rate=1, num_workers=dataloader_num_workers)
train_data = DataLoader(train_ds, batch_size=traning_batch_size, shuffle=True, num_workers=dataloader_num_workers)

valid_ds = CacheDataset(data=valid_data_dicts, transform=val_transforms, cache_rate=1, num_workers=dataloader_num_workers)
valid_data = DataLoader(valid_ds, batch_size=valid_batch_size, num_workers=dataloader_num_workers)

test_ds = CacheDataset(data=test_data_dicts, transform=val_transforms, cache_rate=1, num_workers=dataloader_num_workers)
test_data = DataLoader(test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers)


# ------- Model setting -------
# Initilize Model network
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
#device = torch.device(cfg.CUDA)
cfg.TRAIN.USE_FLIPPED = True
# whether perform class_agnostic bbox regression
fasterRCNN = resnet(all_cls, 101, pretrained=pre_train, class_agnostic=False)
fasterRCNN.create_architecture()
fasterRCNN.cuda()
# multiple GPU setting
mGPUs = mGPU
#mGPUs = True
#fasterRCNN = nn.DataParallel(fasterRCNN)

init_lr = init_lr
optimizer = torch.optim.Adam(fasterRCNN.parameters(), init_lr)
#if vis:
#    thresh = 0.05
#else:
thresh = 0.0
max_per_image = 10

# Setting the chickpoint file path
# file name (time)
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")               
root_logdir = "/tf/jacky831006/faster-rcnn.pytorch-0.4/tfboard"     
logdir = "{}/run-{}/".format(root_logdir, now) 

# tfboard file path
writer = SummaryWriter(logdir)
if not os.path.isdir(logdir):
    os.makedirs(logdir)
# check_point path
check_path = f'/tf/jacky831006/faster-rcnn.pytorch-0.4/training_checkpoints/{now}'
if not os.path.isdir(check_path):
    os.makedirs(check_path)

#print(f'\n Processing fold #{k}', )
# train(model, epochs, optimizer, train_loader, valid_loader, early_stop)
test_model=train(fasterRCNN, epoch, optimizer, train_data, valid_data, early_stop, all_cls)
# Test running
fasterRCNN.load_state_dict(torch.load(f'/tf/jacky831006/faster-rcnn.pytorch-0.4/training_checkpoints/{now}/{best_metric}.pth'))
test_diou = test(fasterRCNN,test_data, all_cls)

# remove dataloader to free memory
del train_ds
del train_data
del valid_ds
del valid_data
del test_ds
del test_data
gc.collect()
accuracy_list.append(best_metric)
test_diou_list.append(test_diou)
file_list.append(now)
epoch_list.append(best_metric_epoch)
print(f'\n Best valid DIOU:{best_metric}, Best test DIOU:{test_diou}')

final_end_time = time.time()
hours, rem = divmod(final_end_time-first_start_time, 3600)
minutes, seconds = divmod(rem, 60)
all_time = "All time:{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print(all_time)
# write some hyperparameter in ori ini
conf['Config_3d'] = {}
conf['Config_3d']['box_regression_loss'] = cfg.TRAIN.BOX_REGRESSION_LOSS
conf['Config_3d']['class_loss'] = cfg.TRAIN.CLASS_LOSS
conf['Config_3d']['nms_iou'] = cfg.TRAIN.NMS_IOU
conf['Data output'] = {}
conf['Data output']['Running time'] = all_time
conf['Data output']['Data file name'] = str(file_list)
# ini write in type need str type
conf['Data output']['Best Valid DIOU'] = str(accuracy_list)
conf['Data output']['Best Test DIOU'] = str(test_diou_list)
conf['Data output']['Best epoch'] = str(epoch_list)
# 先前讀取的東西也會再寫入一次，因此用覆蓋的
with open(cfgpath, 'w') as f:
    conf.write(f)


'''
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# cross_valid 
cross_valid = True
kfold = 5
accuracy_list = []
test_diou_list = []
file_list = []
epoch_list = []

if cross_valid: 
    random.seed(0)
    random_list=random.sample(range(1, 1000), 5)
    first_start_time = time.time()
    for k in range(kfold):
        # set seed 0 
        #train_dic, test_dic = train_test_split(data_dic, test_size=1 - train_ratio, random_state=k)

        #valid_dic, test_dic = train_test_split(test_dic, test_size=test_ratio/(test_ratio + validation_ratio), random_state=k) 
        
        # set test file as fixed
        train_dic, test_dic = train_test_split(data_dic, test_size=test_ratio ,random_state=0)

        train_dic, valid_dic = train_test_split(train_dic, test_size=validation_ratio/(train_ratio + validation_ratio), random_state=k) 

        print(f'\n Train:{len(train_dic)},Valid:{len(valid_dic)},Test:{len(test_dic)}')

        # ------- Dataset & dataloader import -------

        #train_ds = Dataset(data=test_dic, transform=train_transforms)
        # Dataloader set seed
        set_determinism(seed=0)
        train_ds = CacheDataset(data=train_dic, transform=train_transforms, cache_rate=1.0, num_workers=dataloader_num_workers)
        train_data = DataLoader(train_ds, batch_size=8, num_workers=0)

        valid_ds = CacheDataset(data=valid_dic, transform=val_transforms, cache_rate=1.0, num_workers=dataloader_num_workers)
        valid_data = DataLoader(valid_ds, batch_size=1, num_workers=0)

        test_ds = CacheDataset(data=test_dic, transform=val_transforms, cache_rate=1.0, num_workers=dataloader_num_workers)
        test_data = DataLoader(test_ds, batch_size=1, num_workers=0)
            
        # ------- Model setting -------
        # Initilize Model network
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
        class_list = ['__background__','spleen']
        #device = torch.device(cfg.CUDA)
        cfg.TRAIN.USE_FLIPPED = True
        # whether perform class_agnostic bbox regression
        fasterRCNN = resnet(class_list, 101, pretrained=pre_train, class_agnostic=False)
        fasterRCNN.create_architecture()
        fasterRCNN.cuda()
        # multiple GPU setting
        mGPUs = mGPU
        #mGPUs = True
        #fasterRCNN = nn.DataParallel(fasterRCNN)

        init_lr = init_lr
        optimizer = torch.optim.Adam(fasterRCNN.parameters(), init_lr)
        #if vis:
        #    thresh = 0.05
        #else:
        thresh = 0.0
        max_per_image = 10

        # Setting the chickpoint file path
        # file name (time)
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")               
        root_logdir = "/tf/jacky831006/faster-rcnn.pytorch-0.4/tfboard"     
        logdir = "{}/run-{}/".format(root_logdir, now) 

        # tfboard file path
        writer = SummaryWriter(logdir)
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        # check_point path
        check_path = f'/tf/jacky831006/faster-rcnn.pytorch-0.4/training_checkpoints/{now}'
        if not os.path.isdir(check_path):
            os.makedirs(check_path)

        print(f'\n Processing fold #{k}', )
        # train(model, epochs, optimizer, train_loader, valid_loader, early_stop)
        test_model=train(fasterRCNN, epoch, optimizer, train_data, valid_data, early_stop)

        # Test running
        fasterRCNN.load_state_dict(torch.load(f'/tf/jacky831006/faster-rcnn.pytorch-0.4/training_checkpoints/{now}/{best_metric}.pth'))
        test_diou = test(fasterRCNN,test_data)

        # remove dataloader to free memory
        del train_ds
        del train_data
        del valid_ds
        del valid_data
        del test_ds
        del test_data
        gc.collect()
        accuracy_list.append(best_metric)
        test_diou_list.append(test_diou)
        file_list.append(now)
        epoch_list.append(best_metric_epoch)
        print(f'\n Best valid DIOU:{best_metric}, Best test DIOU:{test_diou}')

    final_end_time = time.time()
    hours, rem = divmod(final_end_time-first_start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    all_time = "All time:{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    print(all_time)
    # write some hyperparameter in ori ini
    conf['Config_3d'] = {}
    conf['Config_3d']['box_regression_loss'] = cfg.TRAIN.BOX_REGRESSION_LOSS
    conf['Config_3d']['class_loss'] = cfg.TRAIN.CLASS_LOSS
    conf['Config_3d']['nms_iou'] = cfg.TRAIN.NMS_IOU
    conf['Data output'] = {}
    conf['Data output']['Running time'] = all_time
    conf['Data output']['Data file name'] = str(file_list)
    # ini write in type need str type
    conf['Data output']['Best Valid DIOU'] = str(accuracy_list)
    conf['Data output']['Best Test DIOU'] = str(test_diou_list)
    conf['Data output']['Best epoch'] = str(epoch_list)
    # 先前讀取的東西也會再寫入一次，因此用覆蓋的
    with open(cfgpath, 'w') as f:
        conf.write(f)


'''