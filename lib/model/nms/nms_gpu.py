import torch
import numpy as np
from _ext import nms
import pdb

def nms_gpu(dets, thresh):
	keep = torch.FloatTensor(dets.size(0), 1).zero_().cuda()
	num_out = torch.FloatTensor(1, 1).zero_().cuda()
	nms_overlap_thresh = torch.FloatTensor(1, 1).fill_(thresh).cuda()
	nms.nms_cuda(keep, dets, num_out, nms_overlap_thresh)
	keep = keep[:num_out.int()[0,0]]
	return keep
    # keep = keep[:num_out.int()
    # return list(order[keep])	

