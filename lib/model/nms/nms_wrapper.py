# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
from model.utils.config import cfg
# from model.nms.gpu_nms import gpu_nms
# from model.nms.cpu_nms import cpu_nms
from model.nms.nms_gpu import nms_gpu

import pdb

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
        # ---numpy version---
        # original: return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
        # ---pytorch version---
        return nms_gpu(dets, thresh)
    # else:
    #     return cpu_nms(dets, thresh)
