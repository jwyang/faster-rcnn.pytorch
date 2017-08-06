"""The data layer used during training to train a Fast R-CNN network.
This code is adapt from layer.py to support pytorch multi-thread data loading. 
"""

import torch.utils.data as data
import numpy as np

from model.config import cfg
from datasets.factory import get_imdb
import roi_data_layer.roidb as rdl_roidb

class RoIDataLayer(data.Dataset): # torch wrapper
  """Fast R-CNN data layer used for training."""

  def __init__(self, imdb_name):
    """Set the roidb to be used by this layer during training."""
    imdb, roidb = self.combined_roidb(imdb_name)
    print('{:d} roidb entries'.format(len(roidb)))
    self._roidb = roidb

  def __getitem__(self, index):
    pass

  def __len__(self):
    return len(self._roidb)

  def combined_roidb(self, imdb_names):
    """
    Combine multiple roidbs
    """

    def get_roidb(imdb_name):
      imdb = get_imdb(imdb_name)
      print('Loaded dataset `{:s}` for training'.format(imdb.name))
      imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
      print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
      roidb = get_training_roidb(imdb)
      return roidb

    def get_training_roidb(imdb):
      """Returns a roidb (Region of Interest database) for use in training."""
      if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

      print('Preparing training data...')
      rdl_roidb.prepare_roidb(imdb)
      print('done')

      return imdb.roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
      for r in roidbs[1:]:
        roidb.extend(r)
      tmp = get_imdb(imdb_names.split('+')[1])
      imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
      imdb = get_imdb(imdb_names)
    return imdb, roidb