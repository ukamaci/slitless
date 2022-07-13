""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
from slitless.networks.unet_parts import DoubleConv,Up,Down,OutConv


class UNet(nn.Module):
    def __init__(self, in_channels, start_filters, bilinear=True, residual=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.residual = residual
        sf = start_filters

        self.inc = DoubleConv(in_channels, sf)
        self.down1 = Down(sf, sf*2, (2,2))
        self.down2 = Down(sf*2, sf*4, (2,2))
        self.down3 = Down(sf*4, sf*8, (2,2))
        factor = 2 if bilinear else 1
        self.down4 = Down(sf*8, sf*16 // factor, (2,2))
        self.up1 = Up(sf*16, sf*8 // factor, (2,2), bilinear)
        self.up2 = Up(sf*8, sf*4 // factor, (2,2), bilinear)
        self.up3 = Up(sf*4, sf*2 // factor, (2,2), bilinear)
        self.up4 = Up(sf*2, sf, (2,2), bilinear)
        # self.outc = nn.Conv2d(sf, 1, kernel_size=1)
        self.outc = OutConv(sf, 3)

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
