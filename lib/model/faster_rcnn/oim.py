from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import autograd


class OIM(autograd.Function):
    def __init__(self, lut, queue, momentum):
        super(OIM, self).__init__()
        self.lut = lut
        self.queue = queue
        self.momentum = momentum  # TODO: use exponentially weighted average

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.queue.t())
        # # ======================test=======================
        # for i, (x, y) in enumerate(zip(inputs, targets)):
        #     if y == -1:
        #         tmp = torch.cat((self.queue[1:], x.view(1, -1)), 0)
        #         self.queue[:, :] = tmp[:, :]
        #     elif y < len(self.lut):
        #         self.lut[y] = self.momentum * self.lut[y] + \
        #                       (1. - self.momentum * x)
        #         self.lut[y] /= self.lut[y].norm()
        #     else:
        #         continue
        return torch.cat((outputs_labeled, outputs_unlabeled), 1)

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((self.lut, self.queue), 0))

        for i, (x, y) in enumerate(zip(inputs, targets)):
            if y == -1:
                tmp = torch.cat((self.queue[1:], x.view(1, -1)), 0)
                self.queue[:, :] = tmp[:, :]
            elif y < len(self.lut):
                self.lut[y] = self.momentum * self.lut[y] + \
                              (1. - self.momentum * x)
                self.lut[y] /= self.lut[y].norm()
            else:
                continue

        return grad_inputs, None


def oim(inputs, targets, lut, queue, momentum=0.5):
    return OIM(lut, queue, momentum)(inputs, targets)
