# Install on windows
## Environment required
- Visual Studio 2017
- CUDA 9.1
- Torch 0.4

## Install
- git clone this project
- git checkout windows
- Generate cython_bbox.cp36-win_amd64.pyd and _mask.cp36-win_amd64.pyd
```
 cd lib
python setup.py build_ext --inplace
```
- Open windows/faster_rcnn.sln
- Configured release/x64 for each of project(Debug was not configured)
- Bulid all project in this solution to generate _num.pyd,_roi_pooling.pyd,_roi_crop.pyd and _roi_align.pyd
- Add path of lib to python model environment, or open FasterRCNN.sln and debug with visual studio 2017

## Train and Test and demo
- Please follow the instructions in [README.md](../README.md)


