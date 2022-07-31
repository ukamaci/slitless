""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
from slitless.networks.unet_parts import DoubleConv,Up,Down,OutConv

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, outch_type='all', numlayers=4, start_filters=16, ksizes=[(3,3)], bilinear=True, residual=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.numlayers = numlayers
        self.ksizes = ksizes
        self.bilinear = bilinear
        self.residual = residual
        self.outch_type = outch_type
        sf = start_filters

        if len(ksizes)==1:
            ksizes = ksizes * numlayers

        self.inc = DoubleConv(in_channels, sf, ksize=ksizes[0])
        factor = 2 if bilinear else 1
        for i in range(numlayers):
            if i==numlayers-1:
                setattr(self, f'down{i+1}', Down(sf*2**i, 2*sf*2**i//factor, (2,2), ksize=ksizes[i]))
                setattr(self, f'up{i+1}', Up(sf*2, sf, (2,2), ksize=ksizes[numlayers-i-1]))
                continue
            setattr(self, f'down{i+1}', Down(sf*2**i, 2*sf*2**i, (2,2), ksize=ksizes[i+1]))
            setattr(self, f'up{i+1}', Up(sf*2**(numlayers-i), sf*2**(numlayers-i-1)//factor, (2,2), bilinear, ksize=ksizes[numlayers-i-1]))
        self.outc = OutConv(sf, out_channels)

    def forward(self, x0):
        xs=[self.inc(x0)]
        for i in range(self.numlayers):
            xs.append(getattr(self, f'down{i+1}')(xs[i]))
        x = self.up1(xs[-1], xs[-2])
        for i in range(self.numlayers-1):
            x = getattr(self, f'up{i+2}')(x, xs[-3-i])
        logits = self.outc(x)

        if self.residual:
            return x0 - logits
        else:
            return logits


class UNet_fixed(nn.Module):
    def __init__(self, in_channels, out_channels, outch_type='all', start_filters=16, ksize=(3,3), bilinear=True, residual=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.residual = residual
        self.outch_type = outch_type
        sf = start_filters

        self.inc = DoubleConv(in_channels, sf, ksize=ksize)
        self.down1 = Down(sf, sf*2, (2,2), ksize=ksize)
        self.down2 = Down(sf*2, sf*4, (2,2), ksize=ksize)
        self.down3 = Down(sf*4, sf*8, (2,2), ksize=ksize)
        factor = 2 if bilinear else 1
        self.down4 = Down(sf*8, sf*16 // factor, (2,2), ksize=ksize)
        self.up1 = Up(sf*16, sf*8 // factor, (2,2), bilinear, ksize=ksize)
        self.up2 = Up(sf*8, sf*4 // factor, (2,2), bilinear, ksize=ksize)
        self.up3 = Up(sf*4, sf*2 // factor, (2,2), bilinear, ksize=ksize)
        self.up4 = Up(sf*2, sf, (2,2), bilinear, ksize=ksize)
        # self.outc = nn.Conv2d(sf, 1, kernel_size=1)
        self.outc = OutConv(sf, out_channels)

    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.residual:
            return x0 - logits
        else:
            return logits

