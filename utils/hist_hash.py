# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


import torch
import numpy as np
from scipy.sparse import csr_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HistHash(object):
    def __init__(self, pca_num_filter2, hist_blk_size, hist_blk_over):
        self.pca_num_filter2 = pca_num_filter2
        self.hist_blk_size = hist_blk_size
        self.hist_blk_over = hist_blk_over

    def __call__(self, data):
        map_weights = torch.pow(2.0, torch.arange(
            self.pca_num_filter2-1, -1, -1)).to(DEVICE)
        stride = round((1-self.hist_blk_over)*self.hist_blk_size)
        unfold = torch.nn.Unfold(kernel_size=self.hist_blk_size, stride=stride)

        im_map = torch.heaviside(data, torch.tensor(0.0))
        im_map = im_map.reshape(im_map.shape[0], int(
            im_map.shape[1]/self.pca_num_filter2), self.pca_num_filter2, im_map.shape[2], im_map.shape[3])
        im_map = map_weights[:, None, None] * im_map
        im_map = torch.sum(im_map, dim=2)

        patches = unfold(im_map).squeeze(0)
        patches = patches[None, :, :] if patches.ndim==2 else patches
        patches = patches.permute(0, 2, 1)
        patches = patches.reshape([patches.shape[0], self.pca_num_filter2 * patches.shape[1],
                                   self.hist_blk_size*self.hist_blk_size])
        
        codes = torch.tile(torch.arange(0, 2**self.pca_num_filter2, 1), [patches.shape[2],1]).to(DEVICE)
        features = torch.sum((codes.T[:, None, :] - patches[:, None, :])==0, dim=3)
        features = features.permute(0, 2, 1).reshape(features.shape[0], features.shape[1]*features.shape[2])
        features = csr_matrix(features.cpu())

        return features
