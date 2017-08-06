import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from proposal_layer import _ProposalLayer

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self):
        super(_RPN, self).__init__()

    def forward(self, input):
        return input
