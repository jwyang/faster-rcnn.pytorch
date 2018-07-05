from __future__ import absolute_import
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import numpy as np
from .voc_eval import voc_ap

def vg_eval( detpath,
             gt_roidb,
             image_index,
             classindex,
             ovthresh=0.5,
             use_07_metric=False,
             eval_attributes=False):
    """rec, prec, ap, sorted_scores, npos = voc_eval(
                                detpath, 
                                gt_roidb,
                                image_index,
                                classindex,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the Visual Genome evaluation.

    detpath: Path to detections
    gt_roidb: List of ground truth structs.
    image_index: List of image ids.
    classindex: Category index
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for item,imagename in zip(gt_roidb,image_index):
        if eval_attributes:
            bbox = item['boxes'][np.where(np.any(item['gt_attributes'].toarray() == classindex, axis=1))[0], :]
        else:
            bbox = item['boxes'][np.where(item['gt_classes'] == classindex)[0], :]
        difficult = np.zeros((bbox.shape[0],)).astype(np.bool)
        det = [False] * bbox.shape[0]
        npos = npos + sum(~difficult)        
        class_recs[str(imagename)] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    if npos == 0:
        # No ground truth examples
        return 0,0,0,0,npos

    # read dets
    with open(detpath, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        # No detection examples
        return 0,0,0,0,npos

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = -np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    
    return rec, prec, ap, sorted_scores, npos
