# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='nearest')


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, bias=False, groups=1):
    padding = kernel_size // 2
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups))

class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # x: [B, L, C]
        weights = torch.softmax(self.attn(x), dim=1)  # [B, L, 1]
        return (x * weights).sum(dim=1), weights

class FCNsignalMod(nn.Module):
    """FCNsignal for predicting base-resolution binding signals"""
    def __init__(self, motiflen=12, n_features=0):
        super(FCNsignalMod, self).__init__()
        self.n_features = n_features
         # encoding
        self.conv1 = nn.Conv1d(4 + self.n_features, 64, kernel_size=motiflen, padding=motiflen // 2)
        self.pool1 = nn.MaxPool1d(4, stride=4)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(4, stride=4)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(4, stride=4)
        # decoding
        self.gru1 = nn.GRU(64, 64, batch_first=True)
        self.gru2 = nn.GRU(64, 64, batch_first=True)
        self.attention = Attention(input_dim=128)
        self.blend4 = bn_relu_conv(256, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend1 = bn_relu_conv(64, 1, kernel_size=3)
        # general functions
        self.relu = nn.ELU(alpha=0.1, inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        # encoding
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        res2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1 + res2)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        res3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1 + res3)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        out1 = out1.permute(0, 2, 1)  # [B, L, C]
        out1_1, _ = self.gru1(out1)
        out1_2, _ = self.gru2(torch.flip(out1, [1]))
        out1_2 = torch.flip(out1_2, [1])
        out1 = torch.cat([out1_1, out1_2], dim=-1)  # [B, L, 128]
        out_attn, _ = self.attention(out1)  # [B, 128]
        out1 = out1.permute(0, 2, 1)  # [B, 128, L]
        skip4 = out1
        # decoding
        up4 = upsample(out_attn.unsqueeze(-1), skip4.size(-1)).expand_as(skip4)
        up4 = torch.cat([up4, skip4], dim=1)
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1]) + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1]) + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        out_dense = self.blend1(up1)
        # return out_dense.squeeze(1)
        return F.relu(out_dense.squeeze(1)) + 1e-5
