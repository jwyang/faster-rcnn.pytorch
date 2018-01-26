#!/usr/bin/env bash

cp ../vgg16_caffe.pth ./data/pretrained_model/
cp ../resnet101_caffe.pth ./data/pretrained_model/

cd lib
sh make.sh
cd ../data

ln -s ../../VOCdevkit VOCdevkit2007
ln -s ../../person_search/data/psdb psdb
cd ..
