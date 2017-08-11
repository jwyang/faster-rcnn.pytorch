import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.autograd import Variable

import pdb

class DataParallelModified(nn.Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelModified, self).__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0]) 

    def forward(self, *inputs, **kwargs):
        #pdb.set_trace()
        #inputs = inputs[0]
        inputs, kwargs = self.scatter(inputs[0], kwargs, self.device_ids)
        pdb.set_trace()
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        pdb.set_trace()

        return self.gather(outputs, self.output_device)            

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    """Evaluates module(input) in parallel across the GPUs given in device_ids.
    This is the functional version of the DataParallel module.
    Args:
        module: the module to evaluate in parallel
        inputs: inputs to the module
        device_ids: GPU ids on which to replicate module
        output_device: GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Variable containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)

def scatter_map(inputs, target_gpus):
    input_list = []
    for i in range(len(inputs)):
        var_list = []
        for var in inputs[i]:
            var_list.append(var.cuda(target_gpus[i]))
        input_list.append(tuple(var_list))
    return tuple(input_list)

def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter_map(inputs, target_gpus)
    if kwargs is None or len(kwargs) == 0:
        kwargs = tuple({} for _ in inputs)
    else:
        kwargs = scatter(kwargs, target_gpus, dim)[:len(inputs)]
    return inputs, kwargs