#!/usr/bin/env bash

cp ../vgg16_caffe.pth ./data/pretrained_model/

cd lib
sh make.sh
cd ..

ln -s ../VOCdevkit ./data/VOCdevkit2007