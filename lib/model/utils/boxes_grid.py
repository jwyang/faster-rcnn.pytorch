# --------------------------------------------------------
# Subcategory CNN
# Copyright (c) 2015 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np
import math
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
# <<<< obsolete

def get_boxes_grid(image_height, image_width):
    """
    Return the boxes on image grid.
    calling this function when cfg.IS_MULTISCALE is True, otherwise, calling rdl_roidb.prepare_roidb(imdb) instead.
    """

    # fixed a bug, change cfg.TRAIN.SCALES to cfg.TRAIN.SCALES_BASE
    # coz, here needs a ratio around 1.0, not the accutual size.
    # height and width of the feature map
    if cfg.NET_NAME == 'CaffeNet':
        height = np.floor((image_height * max(cfg.TRAIN.SCALES_BASE) - 1) / 4.0 + 1)
        height = np.floor((height - 1) / 2.0 + 1 + 0.5)
        height = np.floor((height - 1) / 2.0 + 1 + 0.5)

        width = np.floor((image_width * max(cfg.TRAIN.SCALES_BASE) - 1) / 4.0 + 1)
        width = np.floor((width - 1) / 2.0 + 1 + 0.5)
        width = np.floor((width - 1) / 2.0 + 1 + 0.5)
    elif cfg.NET_NAME == 'VGGnet':
        height = np.floor(image_height * max(cfg.TRAIN.SCALES_BASE) / 2.0 + 0.5)
        height = np.floor(height / 2.0 + 0.5)
        height = np.floor(height / 2.0 + 0.5)
        height = np.floor(height / 2.0 + 0.5)

        width = np.floor(image_width * max(cfg.TRAIN.SCALES_BASE) / 2.0 + 0.5)
        width = np.floor(width / 2.0 + 0.5)
        width = np.floor(width / 2.0 + 0.5)
        width = np.floor(width / 2.0 + 0.5)
    else:
        assert (1), 'The network architecture is not supported in utils.get_boxes_grid!'

    # compute the grid box centers
    h = np.arange(height)
    w = np.arange(width)
    y, x = np.meshgrid(h, w, indexing='ij') 
    centers = np.dstack((x, y))
    centers = np.reshape(centers, (-1, 2))
    num = centers.shape[0]

    # compute width and height of grid box
    area = cfg.TRAIN.KERNEL_SIZE * cfg.TRAIN.KERNEL_SIZE
    aspect = cfg.TRAIN.ASPECTS  # height / width
    num_aspect = len(aspect)
    widths = np.zeros((1, num_aspect), dtype=np.float32)
    heights = np.zeros((1, num_aspect), dtype=np.float32)
    for i in xrange(num_aspect):
        widths[0,i] = math.sqrt(area / aspect[i])
        heights[0,i] = widths[0,i] * aspect[i]

    # construct grid boxes
    centers = np.repeat(centers, num_aspect, axis=0)
    widths = np.tile(widths, num).transpose()
    heights = np.tile(heights, num).transpose()

    x1 = np.reshape(centers[:,0], (-1, 1)) - widths * 0.5
    x2 = np.reshape(centers[:,0], (-1, 1)) + widths * 0.5
    y1 = np.reshape(centers[:,1], (-1, 1)) - heights * 0.5
    y2 = np.reshape(centers[:,1], (-1, 1)) + heights * 0.5
    
    boxes_grid = np.hstack((x1, y1, x2, y2)) / cfg.TRAIN.SPATIAL_SCALE

    return boxes_grid, centers[:,0], centers[:,1]
