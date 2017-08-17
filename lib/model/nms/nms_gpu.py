import torch
import numpy as np
from _ext import nms
import pdb

def nms_gpu(dets, thresh):
	keep = torch.Tensor(dets.size(0), 1).zero_().type_as(dets).int()
	num_out = torch.Tensor(1).zero_().type_as(dets).int()
	nms.nms_cuda(keep, dets, num_out, thresh)
	keep = keep[:num_out[0]]
	return keep
    # keep = keep[:num_out.int()
    # return list(order[keep])	

