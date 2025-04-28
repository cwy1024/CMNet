# CMNet
import os
import time
import math
import torch
import joblib
import random
import warnings
import argparse
import numpy as np
import torchvision
import pandas as pd
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import sklearn.externals
import torch.optim as optim
# from dataset import Dataset
from datetime import datetime
from skimage.io import imread
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
from CMNet.CMMNet.Mamba_cnn.Mamba_test import VisionMamba

from CMNet.CMMNet.module.Edge2 import *
from CMNet.CMMNet.Mamba_cnn.Mamba_test import PatchEmbed, create_block


def SeedSed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SeedSed(seed=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


# Mamba

class Downsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, imgsize, patchsize, stride, inchannels, embeddim, self_is=True):
        super(Downsample_block, self).__init__()
        self.is_down = self_is

        # cnn
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

        # vim
        self.PEmbed = PatchEmbed(img_size=imgsize, patch_size=patchsize, stride=stride, in_chans=inchannels,
                                 embed_dim=embeddim)
        deepth = 1
        self.ViMamba1 = nn.ModuleList(
            create_block(d_model=embeddim, ssm_cfg=None) for _ in range(deepth)
        )
        self.conv1x1_final1 = nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3,padding=1)

    def forward(self, x):

        # Vim
        Vim_x = x
        b, c, h, w = Vim_x.shape
        Vim_x = self.PEmbed(Vim_x)
        res1 = None
        for module in self.ViMamba1:
            Dm, res1 = module(Vim_x, res1)
        Vim_x = torch.permute(Vim_x, (0, 2, 1)).reshape(b, c, h, w)
        # cnn
        residual = self.conv1x1(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        y = F.gelu(self.bn2(self.conv2(x)))
        y = y + residual
        y = torch.cat((y, Vim_x), dim=1)
        y = self.conv1x1_final1(y)
        y=y+residual
        if self.is_down:
            x = F.max_pool2d(y, 2, stride=2)
            return x, y
        else:
            return y

class CrossChannelAndCrossSpatialEnhance(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels):
        super(CrossChannelAndCrossSpatialEnhance, self).__init__()

        self.pz1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, dilation=1, padding=1, stride=1)
        self.pzbn1 = nn.BatchNorm2d(in_channels // 2)
        self.pz2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, dilation=2, padding=2, stride=1)
        self.pzbn2 = nn.BatchNorm2d(in_channels // 2)
        self.pz3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, dilation=3, padding=3, stride=1)
        self.pzbn3 = nn.BatchNorm2d(in_channels // 2)
        self.SaSigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1x1 = nn.Conv2d((in_channels // 2) * 3, in_channels // 2, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.linear1 = nn.Linear(in_channels, in_channels // 2)
        self.linear2 = nn.Linear(in_channels // 2, in_channels // 2)
        self.CASigmoid = nn.Sigmoid()

    def forward(self, e1, e2):
        x = torch.cat((e1, e2), dim=1)
        # sa
        Sa = torch.cat((e1, e2), dim=1)
        Sa1 = F.relu(self.pzbn1(self.pz1(Sa)))
        Sa2 = F.relu(self.pzbn2(self.pz2(Sa)))
        Sa3 = F.relu(self.pzbn3(self.pz3(Sa)))
        Sa = self.conv1x1(torch.cat((Sa1, Sa2, Sa3), dim=1))
        Sa = self.SaSigmoid(Sa)

        SAe1 = e1 * Sa
        SAe2 = e2 * Sa
        x = torch.cat((x, SAe1, SAe2), dim=1)
        x = F.relu(self.bn1(self.conv(x)))

        return x

class Upsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, i=0):
        super(Upsample_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        B, C, H, W = y.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = torch.cat((x, y), dim=1)
        residual = self.conv1x1(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x + residual
        return x

class Edge(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels):
        super(Edge, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sobel_x1, self.sobel_y1 = get_sobel(1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.conv1x1(x)
        e = run_sobel(self.sobel_x1, self.sobel_y1, x)

        return e

class Seg_head(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels):
        super(Seg_head, self).__init__()

        self.conv1 = nn.Conv2d(256, 12, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 12, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 12, kernel_size=1)

        self.DConv1 = nn.Conv2d(40, 12, kernel_size=3, dilation=1, padding=1)
        self.DConv2 = nn.Conv2d(12, 12, kernel_size=3, dilation=2, padding=2)
        self.DConv3 = nn.Conv2d(12, 12, kernel_size=3, dilation=3, padding=3)
        self.BN1 = nn.BatchNorm2d(12)

        self.PEmbed1 = PatchEmbed(img_size=224, patch_size=8, stride=8, in_chans=12,
                                  embed_dim=768)
        self.PEmbed2 = PatchEmbed(img_size=224, patch_size=8, stride=8, in_chans=12,
                                  embed_dim=768)
        self.PEmbed3 = PatchEmbed(img_size=224, patch_size=8, stride=8, in_chans=12,
                                  embed_dim=768)

        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)

        self.conv1x1 = nn.Conv2d(24, 3, kernel_size=1)

    def forward(self, out1, out2, out3, edge):
        out1 = F.interpolate(out1, size=(224, 224), mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=(224, 224), mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=(224, 224), mode='bilinear', align_corners=False)

        out1 = self.conv1(out1)
        out2 = self.conv2(out2)
        out3 = self.conv3(out3)
        residual = out3

        out = torch.cat((out1, out2, out3,edge), dim=1)
        out = F.gelu(self.BN1(self.DConv3(self.DConv2(self.DConv1(out)))))

        b, c, h, w = out.shape

        out = self.PEmbed1(out)
        out3 = self.PEmbed2(out3)

        q = self.query(out)
        k = self.key(out)
        v = self.value(out3)

        attention_scores1 = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(768)
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        output1 = torch.matmul(attention_weights1, v)
        output1 = torch.permute(output1, (0, 2, 1)).reshape(b, c, h, w)

        out = torch.cat((residual, output1), dim=1)
        out = self.conv1x1(out)

        return out

class Mamba_Cnn(nn.Module):
    SeedSed(seed=10)

    def __init__(self):
        out_chan = 3
        super(Mamba_Cnn, self).__init__()
        # version 1
        self.down1 = Downsample_block(2, 32,imgsize=224,patchsize=8,stride=8,inchannels=2,embeddim=128)
        self.down2 = Downsample_block(32, 64,imgsize=112,patchsize=4,stride=4,inchannels=32,embeddim=512)
        self.down3 = Downsample_block(64, 128,imgsize=56,patchsize=2,stride=2,inchannels=64,embeddim=256)
        self.down4 = Downsample_block(2, 32,imgsize=224,patchsize=8,stride=8,inchannels=2,embeddim=128)
        self.down5 = Downsample_block(32, 64,imgsize=112,patchsize=4,stride=4,inchannels=32,embeddim=512)
        self.down6 = Downsample_block(64, 128,imgsize=56,patchsize=2,stride=2,inchannels=64,embeddim=256)
        self.bottle1 = Downsample_block(256, 512, self_is=False,imgsize=28,patchsize=1,stride=1,inchannels=256,embeddim=256)

        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)

        # CrossChannelAndCrossSpatialEnhance
        self.CCCAE1 = CrossChannelAndCrossSpatialEnhance(in_channels=64)
        self.CCCAE2 = CrossChannelAndCrossSpatialEnhance(in_channels=128)
        self.CCCAE3 = CrossChannelAndCrossSpatialEnhance(in_channels=256)

        self.edge1 = Edge(32)
        self.edge2 = Edge(64)
        self.edge4 = Edge(32)
        self.edge5 = Edge(64)

        # seg head
        self.seghead = Seg_head(192)

    def forward(self, x):
        chunks = torch.chunk(x, chunks=4, dim=1)
        flair = chunks[0]
        t1 = chunks[1]
        t1ce = chunks[2]
        t2 = chunks[3]
        x1 = torch.cat((t1, t1ce), dim=1)
        x2 = torch.cat((t2, flair), dim=1)

        # Encoder1
        x1, y1_1 = self.down1(x1)
        x1, y1_2 = self.down2(x1)
        x1, y1_3 = self.down3(x1)
        e1_1 = self.edge1(y1_1)
        e1_2 = self.edge2(y1_2)
        # Encoder2
        x2, y2_1 = self.down4(x2)
        x2, y2_2 = self.down5(x2)
        x2, y2_3 = self.down6(x2)
        e2_1 = self.edge4(y2_1)
        e2_2 = self.edge5(y2_2)
        e = torch.cat((e1_1, e1_2, e2_1, e2_2), dim=1)  # 边缘特征整合

        y1 = self.CCCAE1(y1_1, y2_1)
        y2 = self.CCCAE2(y1_2, y2_2)
        y3 = self.CCCAE3(y1_3, y2_3)

        x = torch.cat((x1, x2), dim=1)
        x = self.bottle1(x)

        x = self.up3(x, y3)
        out1 = x
        x = self.up2(x, y2)
        out2 = x
        x = self.up1(x, y1)
        out3 = x

        out = self.seghead(out1, out2, out3, e)

        return out


if __name__ == '__main__':
    SeedSed(seed=10)
    input = torch.randn((1, 4, 224, 224)).to(device)
    model = Mamba_Cnn().to(device)
    out = model(input)
    print(out.shape)
    # print(out)
