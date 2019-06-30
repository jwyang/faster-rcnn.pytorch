from __future__ import absolute_import
import torch
import numpy as np
from ._ext import nms
import pdb

def nms_gpu(dets, thresh):
	keep = dets.new(dets.size(0), 1).zero_().int()
	num_out = dets.new(1).zero_().int()
	nms.nms_cuda(keep, dets, num_out, thresh)
	keep = keep[:num_out[0]]
	return keep
