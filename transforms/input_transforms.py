"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import cv2
import numpy as np

import torch
from torch.nn import functional as F


class ArrayToTensor:
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, imgs):
        imgs
        imgs = [torch.from_numpy(img.transpose((2, 0, 1))).float() for img in imgs]

        return imgs


class Zoom:
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, imgs):
        imgs = [cv2.resize(img, (self.new_w, self.new_h)) for img in imgs]


        return imgs


# We'll keep this function but it won't be used since we're not loading segmentation data
def full_segs_to_adj_maps(full_segs, win_size=9, pad_mode="replicate"):
    """
    Input: full_segs: [B, 1, H, W]
    Output: adj_maps: [B, win_size * win_size, H, W]
    """

    r = (win_size - 1) // 2
    b, _, h, w = full_segs.shape
    full_segs_padded = F.pad(full_segs, (r, r, r, r), mode=pad_mode)

    nb = F.unfold(full_segs_padded, [win_size, win_size])
    nb = nb.reshape((b, win_size * win_size, h, w))

    adj_maps = (full_segs == nb).float()
    return adj_maps
