#!/usr/bin/env python3
# encoding: utf-8



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



'''
    Basic Block     
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class Confidence(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(Confidence, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.FL = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),

            torch.nn.Linear(256, 1),
        )
    def forward(self, x):
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.FL(x)



        return x







class UNet2D(nn.Module):
    """
    2d unet
    Ref:
        3D MRI brain tumor segmentation.
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, in_channels=1, out_channels=1, init_channels=16, p=0.2):
        super(UNet2D, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()


    def make_encoder(self):
        init_channels = self.init_channels

        #self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))
        # self.pool_1 = nn.functional.interpolate(scale_factor=0.5, mode="bilinear")
        # self.pool_2 = nn.functional.interpolate(scale_factor=0.25, mode="bilinear")
        # self.pool_3 = nn.functional.interpolate(scale_factor=0.125, mode="bilinear")

        self.conv1a = nn.Conv2d(self.in_channels, init_channels, 3, padding=1)
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv2d(init_channels, init_channels * 2, 3, stride=2, padding=1)  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv2d(init_channels * 2, init_channels * 4, 3, stride=2, padding=1)

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv2d(init_channels * 4, init_channels * 8, 3, stride=2, padding=1)

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)

        self.ds4 = nn.Conv2d(init_channels * 8, init_channels * 16, 3, stride=2, padding=1)

        self.conv5a = BasicBlock(init_channels * 16, init_channels * 16)
        self.conv5b = BasicBlock(init_channels * 16, init_channels * 16)

        self.conf = Confidence(init_channels * 16, 1)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up5conva = nn.Conv2d(init_channels * 16, init_channels * 8, 1)
        self.up5 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up5convb = BasicBlock(init_channels * 8, init_channels * 8)

        self.up4conva = nn.Conv2d(init_channels * 8, init_channels * 4, 1)
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv2d(init_channels * 4, init_channels * 2, 1)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv2d(init_channels * 2, init_channels, 1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.up1conv = nn.Conv2d(init_channels, self.out_channels, 1)


    def forward(self, x):
        # #print("input - x :", x.shape)
        # x_1 = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear")
        # x_2 = nn.functional.interpolate(x, scale_factor=0.25, mode="bilinear")
        # x_3 = nn.functional.interpolate(x, scale_factor=0.125, mode="bilinear")
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)

        c1d = self.ds1(c1)
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)
        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)

        c4d = self.ds4(c4)
        c5 = self.conv5a(c4d)
        c5d = self.conv5b(c5)

        # conf = self.conf(c5d)

        u5 = self.up5conva(c5d)
        u5 = self.up5(u5)
        u5 = u5 + c4
        u5 = self.up5convb(u5)
        u4 = self.up4conva(u5)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)
        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)
        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)
        uout = self.up1conv(u2)
        uout = torch.sigmoid(uout)
        # conf = torch.sigmoid(conf)
        # print("uout.shape, conf.value:", uout.shape, conf.shape, conf[0], conf)
        uout_1 = nn.functional.interpolate(uout, scale_factor=0.5, mode="bilinear")
        uout_2 = nn.functional.interpolate(uout, scale_factor=0.25, mode="bilinear")
        uout_3 = nn.functional.interpolate(uout, scale_factor=0.125, mode="bilinear")
        return uout, uout_1, uout_2, uout_3