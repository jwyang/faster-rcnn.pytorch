# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
from model.utils.config import cfg
from model.nms.cpu_nms import cpu_nms
from model.nms.nms_gpu import nms_gpu

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
        # ---numpy version---
        # original: return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
        # ---pytorch version---
        return nms_gpu(dets, thresh)
    else:
        keep = cpu_nms(dets.numpy(), thresh)

        return torch.Tensor(keep)