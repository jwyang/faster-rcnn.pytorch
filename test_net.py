# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import cPickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.minibatch import _get_image_blob
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, \
    get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import pdb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='test dataset',
                        default='psdb', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="./output/trained",
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default=True,
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel,' +
                             ' 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=5, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=22411, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print(
            "WARNING: You have a CUDA device," +
            " so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]']

    elif args.dataset == "psdb":
        args.imdb_name = 'psdb_train'
        args.imdbval_name = 'psdb_test'
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS',
                         '[0.5,1,2]']

    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name,
                                                          False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory' + \
                        ' for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(
                                 args.checksession, args.checkepoch,
                                 args.checkpoint))

    # initilize the network here.
    # add `query` parameter for every query net
    if args.net == 'vgg16':
        query_net = vgg16(imdb.classes, pretrained=False,
                          class_agnostic=args.class_agnostic, query=True)
        gallery_net = vgg16(imdb.classes, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        query_net = resnet(imdb.classes, 101, pretrained=False,
                           class_agnostic=args.class_agnostic,
                           training=False, query=True)
        gallery_net = resnet(imdb.classes, 101, pretrained=False,
                             class_agnostic=args.class_agnostic,
                             training=False)
    elif args.net == 'res50':
        query_net = resnet(imdb.classes, 50, pretrained=False,
                           class_agnostic=args.class_agnostic, query=True)
        gallery_net = resnet(imdb.classes, 50, pretrained=False,
                             class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        query_net = resnet(imdb.classes, 152, pretrained=False,
                           class_agnostic=args.class_agnostic, query=True)
        gallery_net = resnet(imdb.classes, 152, pretrained=False,
                             class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    query_net.create_architecture()
    gallery_net.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    # lut and queue are not needed during testing
    if 'lut' in checkpoint['model'] and 'queue' in checkpoint['model']:
        del checkpoint['model']['lut']
        del checkpoint['model']['queue']

    query_net.load_state_dict(checkpoint['model'])
    gallery_net.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        query_net.cuda()
        gallery_net.cuda()

    # #######################################################################
    # ===========================test for gallery============================
    # #######################################################################

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [0 for _ in xrange(num_images)]
    all_features = [0 for _ in range(num_images)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    gallery_net.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):

        data = data_iter.next()
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, reid_feat, cls_prob, bbox_pred= gallery_net(im_data, im_info,
                                                          gt_boxes, num_boxes)
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        features = reid_feat.data

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        j = 1
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS).view(-1).long()
            cls_dets = cls_dets[keep]
            if vis:
                   im2show = vis_detections(im2show, imdb.classes[j],
                                            cls_dets.cpu().numpy(), 0.3)
            all_boxes[i] = cls_dets.cpu().numpy()
            all_features[i] = features[inds][keep].cpu().numpy()
        else:
            all_boxes[i] = empty_array
            all_features[i] = None

        # TODO: test this function
        # # Limit to max_per_image detections *over all classes*
        # if max_per_image > 0:
        #     image_scores = np.hstack([all_boxes[i][:, -1]])
        #     if len(image_scores) > max_per_image:
        #         image_thresh = np.sort(image_scores)[-max_per_image]
        #         keep = np.where(all_boxes[i][:, -1] >= image_thresh)[0]
        #         all_boxes[i] = all_boxes[i][keep, :]
        #         all_features['feat'][i] = all_features['feat'][i][keep]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        # Do not show flush here!
        # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #                  .format(i + 1, num_images, detect_time, nms_time))
        # sys.stdout.flush()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'
              .format(i + 1, num_images, detect_time, nms_time))

        if vis:
            cv2.imwrite('result.png', im2show)
            pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    del gallery_net

    # #######################################################################
    # ===========================test for query============================
    # #######################################################################

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    num_querys = len(imdb.probes)
    query_features = [0 for _ in xrange(num_querys)]

    # timers
    _t_ = {'query_exfeat': time.time()}
    query_net.eval()

    for i in xrange(num_querys):
        im_name, roi = imdb.probes[i]
        im = [{'image': im_name, 'flipped': False}]
        roi = np.hstack([np.array([[0]]), roi.reshape(1, 4)])

        ex_feat_tic = time.time()

        im_blob, im_scales = _get_image_blob(im, [0])
        roi = roi * im_scales[0]  # very important!!!
        im_info_ = np.array([[im_blob.shape[1], im_blob.shape[2],
                              im_scales[0]]], dtype=np.float32)

        # TODO: maybe some problems here
        data = [torch.from_numpy(im_blob), torch.from_numpy(im_info_),
                torch.from_numpy(roi), torch.from_numpy(np.array([1]))]
        data[0] = data[0].permute(0, 3, 1, 2).contiguous().view(
            3, data[0].size(1), data[0].size(2)).unsqueeze(0)

        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        rois, q_feats, cls_p, bbox_p = query_net(im_data, im_info, gt_boxes,
                                                 num_boxes)
        query_features[i] = q_feats[0].data.cpu().numpy()

        ex_feat_toc = time.time()
        ex_time = ex_feat_toc - ex_feat_tic
        print('query_exfeat: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                       num_querys, ex_time))

    del query_net

    # evaluations
    imdb.evaluate_detections(all_boxes, det_thresh=0.5)
    imdb.evaluate_detections(all_boxes, det_thresh=0.5, labeled_only=True)
    imdb.evaluate_search(all_boxes, all_features, query_features,
                         det_thresh=0.5, gallery_size=100, dump_json=None)

    end = time.time()
    print("test time: %0.4fs" % (end - start))
