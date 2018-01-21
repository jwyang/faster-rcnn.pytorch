from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.imagenet
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import pdb
import pickle
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class imagenet(imdb):
    def __init__(self, image_set, devkit_path, data_path):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = data_path
        synsets_image = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_det.mat'))
        synsets_video = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_vid.mat'))
        self._classes_image = ('__background__',)
        self._wnid_image = (0,)

        self._classes = ('__background__',)
        self._wnid = (0,)

        for i in xrange(200):
            self._classes_image = self._classes_image + (synsets_image['synsets'][0][i][2][0],)
            self._wnid_image = self._wnid_image + (synsets_image['synsets'][0][i][1][0],)

        for i in xrange(30):
            self._classes = self._classes + (synsets_video['synsets'][0][i][2][0],)
            self._wnid = self._wnid + (synsets_video['synsets'][0][i][1][0],)

        self._wnid_to_ind_image = dict(zip(self._wnid_image, xrange(201)))
        self._class_to_ind_image = dict(zip(self._classes_image, xrange(201)))

        self._wnid_to_ind = dict(zip(self._wnid, xrange(31)))
        self._class_to_ind = dict(zip(self._classes, xrange(31)))

        #check for valid intersection between video and image classes
        self._valid_image_flag = [0]*201

        for i in range(1,201):
            if self._wnid_image[i] in self._wnid_to_ind:
                self._valid_image_flag[i] = 1

        self._image_ext = ['.JPEG']

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), 'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data', self._image_set, index + self._image_ext[0])
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt

        if self._image_set == 'train':
            image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr.txt')
            image_index = []
            if os.path.exists(image_set_file):
                f = open(image_set_file, 'r')
                data = f.read().split()
                for lines in data:
                    if lines != '':
                        image_index.append(lines)
                f.close()
                return image_index

            for i in range(1,200):
                print(i)
                image_set_file = os.path.join(self._data_path, 'ImageSets', 'DET', 'train_' + str(i) + '.txt')
                with open(image_set_file) as f:
                    tmp_index = [x.strip() for x in f.readlines()]
                    vtmp_index = []
                    for line in tmp_index:
                        line = line.split(' ')
                        image_list = os.popen('ls ' + self._data_path + '/Data/DET/train/' + line[0] + '/*.JPEG').read().split()
                        tmp_list = []
                        for imgs in image_list:
                            tmp_list.append(imgs[:-5])
                        vtmp_index = vtmp_index + tmp_list

                num_lines = len(vtmp_index)
                ids = np.random.permutation(num_lines)
                count = 0
                while count < 2000:
                    image_index.append(vtmp_index[ids[count % num_lines]])
                    count = count + 1

            for i in range(1,201):
                if self._valid_image_flag[i] == 1:
                    image_set_file = os.path.join(self._data_path, 'ImageSets', 'train_pos_' + str(i) + '.txt')
                    with open(image_set_file) as f:
                        tmp_index = [x.strip() for x in f.readlines()]
                    num_lines = len(tmp_index)
                    ids = np.random.permutation(num_lines)
                    count = 0
                    while count < 2000:
                        image_index.append(tmp_index[ids[count % num_lines]])
                        count = count + 1
            image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr.txt')
            f = open(image_set_file, 'w')
            for lines in image_index:
                f.write(lines + '\n')
            f.close()
        else:
            image_set_file = os.path.join(self._data_path, 'ImageSets', 'val.txt')
            with open(image_set_file) as f:
                image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        filename = os.path.join(self._data_path, 'Annotations', self._image_set, index + '.xml')

        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

if __name__ == '__main__':
    d = datasets.imagenet('val', '')
    res = d.roidb
    from IPython import embed; embed()
