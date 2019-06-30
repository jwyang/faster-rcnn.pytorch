# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

with open('requirements.txt') as inp:
    requirements = inp.read().splitlines()

def get_extensions():
    this_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faster_rcnn', 'lib')
    extensions_dir = os.path.join(this_dir, "model", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "faster_rcnn.lib.model._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_scripts_dict():
    """Construct the scripts dictionary, ie the dictionary of the executables
    """
    scripts_dict = {}
    for root, _, files in os.walk(os.path.join('faster_rcnn', 'scripts')):
        for fil in files:
            if fil.endswith('.py') and fil != '__init__.py':
                path = os.path.join(root, fil)
                module = 'faster_rcnn' + '.'.join(path.split(
                    'faster_rcnn')[-1][:-3].split(os.sep)) + ":main"
                scripts_dict[fil[:-3]] = module
    return scripts_dict


setup(
    name="faster_rcnn",
    version="0.2",
    description="object detection in pytorch",
    install_requires=requirements,
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    entry_points={
        'console_scripts': [
            "faster_rcnn_" + script +
            '=' + d[script] for d in [get_scripts_dict()] for script in d]}
)
