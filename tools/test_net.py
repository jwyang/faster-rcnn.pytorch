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
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms

import pdb

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=700, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10, type=int)  
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      nargs=argparse.REMAINDER)  
  parser.add_argument('--ngpu', dest='ngpu',
                      help='number of gpu',
                      default=1, type=int)


  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
use_multiGPU = True

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def im_detect(net, image):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data, im_scales = net.get_image_blob(image)
    im_info = np.array(
        [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        dtype=np.float32)


    rois, cls_prob, bbox_pred = net(im_data, im_info)
    scores = cls_prob.data.cpu().numpy()
    boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, image.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes

def bbox_transform_inv(boxes, deltas):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = np.exp(dw) * widths.unsqueeze(2)
    pred_h = np.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb = combined_roidb(args.imdb_name)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network')
  load_name = os.path.join(input_dir, 'faster_rcnn_{}.pth'.format(args.checkpoint))


  dataset = roibatchLoader(roidb, imdb.num_classes, False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.ngpu > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.ngpu > 0:
    cfg.CUDA = True

  # initilize the network here.
  fasterRCNN = _fasterRCNN(args.net, imdb.classes)
  params = list(fasterRCNN.parameters())
  print(params[20].sum())
  # load net
  fasterRCNN = torch.load(load_name)
  print(params[20].sum())

  exit()
  print('load model successfully!')

  if args.ngpu > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()

  max_per_image = 300
  thresh = 0.05
  vis = True

  save_name = 'faster_rcnn_10'

  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #    all_boxes[cls][image] = N x 5 array of detections in
  #    (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)

  data_iter = iter(dataloader)

  # timers
  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  for i in range(num_images):

      data = data_iter.next()
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()

      rois, cls_prob, bbox_pred, rpn_loss, rcnn_loss = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois[:, :, 1:5] / data[1][0][2]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          pred_boxes = bbox_transform_inv(boxes, box_deltas)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      # _t['im_detect'].tic()
      det_toc = time.time()      
      detect_time = det_toc - det_tic

      misc_tic = time.time()

      if vis:
          # im2show = np.copy(im[:, :, (2, 1, 0)])
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)          
          # im2show = np.copy(im[:, :, (2, 1, 0)])

          # im2show = np.copy(im_data.data.cpu().squeeze().numpy())
          # im2show = im2show.transpose(1, 2, 0)

      # skip j = 0, because it's the background class
      for j in xrange(1, imdb.num_classes):
          num_candidate = torch.sum(scores[:, j] > thresh)
          if num_candidate == 0:
            continue
          inds = torch.nonzero(scores[:, j] > thresh)
          inds = inds.squeeze()
          cls_scores = scores[:, j][inds]
          cls_boxes = pred_boxes[:, j * 4:(j + 1) * 4][inds]

          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
          keep = nms(cls_dets, cfg.TEST.NMS)
          keep = keep.squeeze().long()

          cls_dets = cls_dets[keep, :]
          cls_dets = cls_dets.cpu().numpy()          
          if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets, 0.1)
          all_boxes[j][i] = cls_dets
      
      # Limit to max_per_image detections *over all classes*
      # if max_per_image > 0:
      #     image_scores = np.hstack([all_boxes[j][i][:, -1]
      #                               for j in xrange(1, imdb.num_classes)])
      #     if len(image_scores) > max_per_image:
      #         image_thresh = np.sort(image_scores)[-max_per_image]
      #         for j in xrange(1, imdb.num_classes):
      #             keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
      #             all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
          .format(i + 1, num_images, detect_time, nms_time))

      # pdb.set_trace()
      if vis:
          cv2.imshow('test', im2show)
          cv2.waitKey(0)
      # pdb.set_trace()

  with open(det_file, 'wb') as f:
      cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

  pdb.set_trace()

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()

  print("test time: %0.4fs" % (end - start))
