# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


import torch
import torch.nn as nn
from torch.nn import parameter
import numpy as np
from utils import HistHash


class RNPCANet(nn.Module):
    def __init__(self, device, in_channels, kernel_size1, kernel_size2, rff_num_filter1, 
                 rff_num_filter2, pca_num_filter1, pca_num_filter2, hist_blk_size,
                 hist_blk_over, sigma1, sigma2):
        super(RNPCANet, self).__init__()

        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.rff_num_filter1 = rff_num_filter1
        self.rff_num_filter2 = rff_num_filter2
        self.pca_num_filter1 = pca_num_filter1
        self.pca_num_filter2 = pca_num_filter2

        self.hist_hash = HistHash(
            device, pca_num_filter2, hist_blk_size, hist_blk_over)

        self.mean_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size1,
            padding=round((kernel_size1-1)/2),
            bias=False
        ).requires_grad_(False)

        nn.init.constant_(self.mean_conv1.weight, 1.0 /
                          (kernel_size1**2 * in_channels))

        self.mean_mul1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rff_num_filter1,
            kernel_size=1,
            bias=False
        ).requires_grad_(False)

        self.rff1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rff_num_filter1,
            kernel_size=kernel_size1,
            padding=round((kernel_size1-1)/2)
        ).requires_grad_(False)

        self.pca1 = nn.Linear(
            in_features=rff_num_filter1,
            out_features=pca_num_filter1,
            bias=False
        ).requires_grad_(False)

        self.mean_conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size2,
            padding=round((kernel_size2-1)/2),
            bias=False
        ).requires_grad_(False)

        nn.init.constant_(self.mean_conv2.weight, 1.0/(kernel_size2**2))

        self.mean_mul2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rff_num_filter2,
            kernel_size=1,
            bias=False
        ).requires_grad_(False)

        self.rff2 = nn.Conv2d(
            in_channels=1,
            out_channels=rff_num_filter2,
            kernel_size=kernel_size2,
            padding=round((kernel_size2-1)/2)
        ).requires_grad_(False)

        self.pca2 = nn.Linear(
            in_features=rff_num_filter2,
            out_features=pca_num_filter2,
            bias=False
        ).requires_grad_(False)

    def forward(self, x, return_loc=None):

        batch_size = x.shape[0]
        # only if the size of the image won't change
        dim1, dim2 = x.shape[2], x.shape[3]

        x_mean = self.mean_conv1(x)
        self.mean_mul1.weight = parameter.Parameter(torch.sum(
            torch.sum(self.rff1.weight, dim=3), dim=2).unsqueeze(2).unsqueeze(3))
        x_mean = self.mean_mul1(x_mean)  # perform w^T.E[x]

        x = self.rff1(x)
        x = x - x_mean  # mean removal
        x = np.sqrt(2/self.rff_num_filter1) * torch.cos(x)

        if return_loc=='rff1':
            return x

        # prepration for projection for the pca layer
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, dim1*dim2, self.rff_num_filter1)
        x = self.pca1(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, self.pca_num_filter1, dim1, dim2)

        # a trick based on creating minibatch to convolve with each channel
        x = x.reshape(batch_size*self.pca_num_filter1, 1, dim1, dim2)

        x_mean = self.mean_conv2(x)
        self.mean_mul2.weight = parameter.Parameter(torch.sum(
            torch.sum(self.rff2.weight, dim=3), dim=2).unsqueeze(2).unsqueeze(3))
        x_mean = self.mean_mul2(x_mean)  # perform w^T.E[x]

        x = self.rff2(x)
        x = x - x_mean # mean removal
        x = np.sqrt(2/self.rff_num_filter2) * torch.cos(x)

        if return_loc=='rff2':
            return x

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], dim1*dim2, self.rff_num_filter1)
        x = self.pca2(x)

        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, self.pca_num_filter2**2, dim1, dim2)

        x = self.hist_hash(x)

        return x
