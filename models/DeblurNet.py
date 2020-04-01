#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
from models.submodules import *
from models.FAC.kernelconv2d import KernelConv2D

from torch import nn

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        #############################
        # Deblurring Branch
        #############################
        # encoder
        ks = 3
        ks_2d = 5
        ch1 = 32
        ch2 = 64
        ch3 = 128
        self.fea = conv(2*ch3, ch3, kernel_size=ks, stride=1)

        self.conv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.conv1_3 = resnet_block(ch1, kernel_size=ks)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.conv2_3 = resnet_block(ch2, kernel_size=ks)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.conv3_3 = resnet_block(ch3, kernel_size=ks)

        self.kconv_warp = KernelConv2D.KernelConv2D(kernel_size=ks_2d)
        self.kconv_deblur = KernelConv2D.KernelConv2D(kernel_size=ks_2d)

        # decoder
        self.upconv2_u = upconv(2*ch3, ch2)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks)

        self.img_prd = conv(ch1, 3, kernel_size=ks)

        #############################
        # Kernel Prediction Branch
        #############################

        # kernel network
        self.kconv1_1 = conv(9, ch1, kernel_size=ks, stride=1)
        self.kconv1_2 = resnet_block(ch1, kernel_size=ks)
        self.kconv1_3 = resnet_block(ch1, kernel_size=ks)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.kconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.kconv2_3 = resnet_block(ch2, kernel_size=ks)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.kconv3_2 = resnet_block(ch3, kernel_size=ks)
        self.kconv3_3 = resnet_block(ch3, kernel_size=ks)

        self.fac_warp = nn.Sequential(
            conv(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

        self.kconv4 = conv(ch3 * ks_2d ** 2, ch3, kernel_size=1)

        self.fac_deblur = nn.Sequential(
            conv(2*ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

    def forward(self, img_blur, last_img_blur, output_last_img, output_last_fea):
        merge = torch.cat([img_blur, last_img_blur, output_last_img], 1)

        #############################
        # Kernel Prediction Branch
        #############################
        # kernel network
        kconv1 = self.kconv1_3(self.kconv1_2(self.kconv1_1(merge)))
        kconv2 = self.kconv2_3(self.kconv2_2(self.kconv2_1(kconv1)))
        kconv3 = self.kconv3_3(self.kconv3_2(self.kconv3_1(kconv2)))
        # fac
        kernel_warp = self.fac_warp(kconv3)
        kconv4 = self.kconv4(kernel_warp)
        kernel_deblur = self.fac_deblur(torch.cat([kconv3, kconv4],1))

        #############################
        # Deblurring Branch
        #############################
        # encoder blur
        conv1_d = self.conv1_1(img_blur)
        conv1_d = self.conv1_3(self.conv1_2(conv1_d))

        conv2_d = self.conv2_1(conv1_d)
        conv2_d = self.conv2_3(self.conv2_2(conv2_d))

        conv3_d = self.conv3_1(conv2_d)
        conv3_d = self.conv3_3(self.conv3_2(conv3_d))

        conv3_d_k = self.kconv_deblur(conv3_d, kernel_deblur)

        # encoder last_clear
        if output_last_fea is None:
            output_last_fea = torch.cat([conv3_d, conv3_d],1)

        output_last_fea = self.fea(output_last_fea)

        conv_a_k = self.kconv_deblur(output_last_fea, kernel_warp)

        conv3 = torch.cat([conv3_d_k, conv_a_k],1)

        # decoder
        upconv2 = self.upconv2_1(self.upconv2_2(self.upconv2_u(conv3)))
        upconv1 = self.upconv1_1(self.upconv1_2(self.upconv1_u(upconv2)))
        output_img = self.img_prd(upconv1) + img_blur
        output_fea = conv3

        return output_img, output_fea
