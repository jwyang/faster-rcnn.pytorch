from __future__ import print_function
import os
import sys
import torch
from torch.utils.ffi import create_extension

#this_file = os.path.dirname(__file__)
torch_root = os.path.join(os.path.dirname(sys.executable),
                          'Lib/site-packages/torch/lib')
cuda_root = os.environ['CUDA_PATH']

sources = ['src/roi_crop.cpp']
headers = ['src/roi_crop.h']
include_dirs = []
defines = []
with_cuda = False

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_crop_cuda.cpp']
    headers += ['src/roi_crop_cuda.h']
    include_dirs += [os.path.join(cuda_root,"include"), 
                     os.path.join(torch_root,'include')]
    defines += [('WITH_CUDA', None)]
    with_cuda = True

extra_objects = ['src/roi_crop_cuda_kernel.lib']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
extra_objects += [os.path.join(torch_root,'ATen.lib'),
                  os.path.join(cuda_root,'lib/x64/cudart.lib'),
                  os.path.join(torch_root,'_C.lib')]
print(extra_objects)

ffi = create_extension(
    '_ext.roi_crop',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    include_dirs=include_dirs
)

if __name__ == '__main__':
    ffi.build()
