import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms.functional import crop
import matplotlib.pyplot as plt


name = 'EEGNet'

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=0)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), bias=bias)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DepthwiseConv2d(nn.Conv2d):
    def __init__(self, F1, D, C, max_norm_val=1, eps=0.01):
        super(DepthwiseConv2d, self).__init__(F1, F1 * D, (C, 1), groups=F1)
        self.max_norm_val = max_norm_val
        self.eps = eps
    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self.max_norm_val)
        return w * (desired / (self.eps + norm))
    def forward(self, input):
        return F.conv2d(input, self._max_norm(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
class Dense(nn.Linear):
    def __init__(self, input_size=16, output_size=2, max_norm_val=0.25, eps=0.01):
        super(Dense, self).__init__(input_size, output_size)
        self.max_norm_val = max_norm_val
        self.eps = eps
    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self.max_norm_val)
        return w * (desired / (self.eps + norm))
    def forward(self, input):
        return F.linear(input, self._max_norm(self.weight), self.bias)
    
class EEGNet(nn.Module):
    '''a comment'''
    def __init__(self, config, output_dim=5, n_electrodes=64, device=None):
        super(EEGNet, self).__init__()
        
        # set device
        if device is None: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device

        self.C = n_electrodes
        self.T = int(config['sampling_frequency'] * config['window_length'] / 1000)
        self.F1 = config['num_temporal_filters']
        self.D = config['num_spatial_filters']
        self.F2 = self.D * self.F1                                # F2: number of pointwise filters; not necessary to be D * F1.
        self.N = output_dim

        block1 = config['block1']
        block2 = config['block2']

        dense_input_size = self.F2 * (((self.T // block1['avg_pool'][1]) - block2['sep_conv'][1] + 1) // block2['avg_pool'][1])

        self._name = "EEGNet-" + str(self.F1) + "," + str(self.D)

        # Block 1
        # -------
        self._conv1 = nn.Conv2d(1, out_channels = self.F1, kernel_size = tuple(block1['conv']), padding='same').to(self.device)
        self._batchnorm1 = nn.BatchNorm2d(self.F1, False).to(self.device)
        self._depthwise = DepthwiseConv2d(self.F1, self.D, self.C, block1['max_norm_value'], block1['eps']).to(self.device)  
        self._batchnorm2 = nn.BatchNorm2d(self.F1 * self.D).to(self.device)
        self._avg_pool1 = nn.AvgPool2d(block1['avg_pool']).to(self.device)
        self._dropout1 = nn.Dropout(block1['dropout']).to(self.device)

        # Block 2
        # -------
        self._seperable = SeparableConv2d(self.F1 * self.D, self.F2, block2['sep_conv']).to(self.device)
        self._batchnorm3 = nn.BatchNorm2d(self.F2).to(self.device)
        self._avg_pool2 = nn.AvgPool2d(block2['avg_pool']).to(self.device)
        self._dropout2 = nn.Dropout(block2['dropout']).to(self.device)

        # Classifier
        # ----------
        self._dense = Dense(dense_input_size, self.N, block2['max_norm_value'], block2['eps']).to(self.device)

    def forward(self, x):
        x = torch.FloatTensor(x) if isinstance(x, np.ndarray) else x
        X = self.reshape_input(x)

        # Block 1
        # -------
        X = self._conv1(X)
        X = self._batchnorm1(X)
        X = self._depthwise(X)
        X = self._batchnorm2(X)
        X = nn.ELU()(X)
        X = self._avg_pool1(X)
        X = self._dropout1(X)

        # Block 2
        # -------
        X = self._seperable(X)
        X = self._batchnorm3(X)
        X = nn.ELU()(X)
        X = self._avg_pool2(X)
        X = self._dropout2(X)
        X = torch.flatten(X, start_dim=1)

        # Classifier
        # ----------
        X = self._dense(X)

        return X

    def reshape_input(self, x):
        '''
        x will be shape (latency/10, n_electrodes)
        Reshapes X to (1, 1, n_electrodes, latency/10) which is batch_size, input_channels, electrodes, timesteps

        at inference the batch size will be 1
        the input_channels is always 1 since EEG is interpreted as a greyscale image
        '''
        if len(x.shape) == 4: #used in training
            x = x.permute(0,3,1,2)
        
        if len(x.shape) == 2:   # used in closed loop, x is (latency/10, channels)
            x = torch.transpose(x, 0, 1)   # x is now (channels, latency)
            x = torch.unsqueeze(torch.unsqueeze(x, 0), 0) # (1, 1, n_electrodes, latency/10)   # use for online

        return x

    def get_name(self):
        return self._name
