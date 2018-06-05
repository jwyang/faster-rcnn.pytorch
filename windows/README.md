# Install on windows
## Environment required
- Visual Studio 2017
- CUDA 9.1
- Torch 0.4

## Install
- git clone this project
- Generate cython_bbox.cp36-win_amd64.pyd and _mask.cp36-win_amd64.pyd
```
 cd lib
python setup.py build_ext --inplace
```
- Open windows/faster_rcnn.sln
- Select release for each project(Debug was not configured)
- Bulid all project in this solution


