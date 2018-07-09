#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

export CUDA_PATH=/usr/local/cuda/

python setup.py build_ext --inplace
rm -rf build

CUDA_ARCH="-gencode arch=compute_61,code=sm_61"

# clean build file
rm model/nms/src/*.o
rm model/roi_pooling/src/*.o
rm model/roi_align/src/*.o
rm model/roi_crop/src/*.o
rm model/psroi_pooling/src/cuda/*.o


# compile NMS
cd model/nms/src
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH

cd ../
python build.py

# compile roi_pooling
cd ../../
cd model/roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile roi_align
cd ../../
cd model/roi_align/src
echo "Compiling roi align kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile roi_crop
cd ../../
cd model/roi_crop/src
echo "Compiling roi crop kernels by nvcc..."
nvcc -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile roi_crop
cd ../../
cd model/psroi_pooling/src/cuda
echo "Compiling psroi pooling kernels by nvcc..."
nvcc -c -o psroi_pooling.cu.o psroi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../../
python build.py
