import torch
import torch.nn as nn
# from ops import *
from Net.UNet_class import *

""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.actfunction = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.actfunction(logits)
        return logits


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


ker = 8

class SearchTransfer2_tmp(nn.Module):
    def __init__(self, ind): # ind = 64
        super(SearchTransfer2_tmp, self).__init__()
        dim = 64
        self.encoder = Encoder1Conv(ind, ind)
        self.encoder1_k = Encoder1Conv(ind, ind)
        self.encoder1_v = Encoder1Conv(ind, ind)

        self.fc_q = nn.Linear(ker * ker * ind, 1024)    # = 4096,1024
        self.bn_q = nn.BatchNorm1d(1024)

        self.fc_k = nn.Linear(ker * ker * ind, 1024)    # = 4096,1024
        self.bn_k = nn.BatchNorm1d(1024)
        self.feed = FeedForward(ind, ind * 4)

    def forward(self, feat):
        feature1_k = self.encoder1_k(feat)
        feature1_v = self.encoder1_v(feat)
        refer = self.encoder(feat)

        print('\n\nline86')
        print('type(feat)      ',type(feat))
        print('type(feature1_k)',type(feature1_k))
        print('feat.size       ',feat.shape)        # 1,64,1024,1024
        print('feature1_k.size ',feature1_k.shape)  # 1,64,1024,1024
        print('feature1_v.size ',feature1_v.shape)  # 1,64,1024,1024
        print('refer.size      ',refer.shape)       # 1,64,1024,1024

        ### search
        q_un = F.unfold(refer, kernel_size=(ker, ker), stride=ker).transpose(1, 2)  # [N, Hr*Wr, C*k*k]
        k_un = F.unfold(feature1_k, kernel_size=(ker, ker), stride=ker).transpose(1, 2)
        v_un = F.unfold(feature1_v, kernel_size=(ker, ker), stride=ker).transpose(1, 2)

        print('\nq_un.size',q_un.shape)     # 1,16384,4096
        print('k_un.size',k_un.shape)       # 1,16384,4096
        print('v_un.size',v_un.shape)       # 1,16384,4096

        q_un = self.bn_q(self.fc_q(q_un).transpose(1, 2)).transpose(1, 2)
        k_un = self.bn_k(self.fc_k(k_un).transpose(1, 2)).transpose(1, 2)
        print('\nq_un.size',q_un.shape)     # 1,16384,1024 =
        print('k_un.size',k_un.shape)       # 1,16384,1024

        k_un = k_un.permute(0, 2, 1)
        print('\nk_un.size',k_un.shape) # 1,1024,16384
        R_lv3 = torch.bmm(q_un, k_un)
        print('R_lv3.size',R_lv3.shape) # 1,16384,16384
        print('q_un.size(2) ** 0.5.size=',q_un.size()) #1,16384,1024
        print('q_un.size(2) ** 0.5.size=',q_un.size(2)) # 1024
        print('q_un.size(2) ** 0.5.size=',q_un.size(2) ** 0.5) # 32
        print('R_lv3[0,0,0]',R_lv3[0,0,0]) # -208.42
        R_lv3 = R_lv3.div(q_un.size(2) ** 0.5)
        print('R_lv3.size',R_lv3.shape) #1,16384,16384
        print('R_lv3[0,0,0]',R_lv3[0,0,0]) # -6.513

        attn = F.softmax(R_lv3, dim=2)
        print('\nattn.size',attn.shape) #1,16384,16384
        output = torch.bmm(attn, v_un).transpose(1, 2)
        print('output.size',output.shape)   #1,4096,16384
        output = F.fold(output, output_size=feat.size()[-2:], kernel_size=(ker, ker), stride=ker)
        print('output.size',output.shape)   #64,1024,1024
        output = self.feed(output) + feat
        print('output.size',output.shape)   #64,1024,1024

        return output



class SearchTransfer2(nn.Module):
    def __init__(self, ind):
        super(SearchTransfer2, self).__init__()
        dim = 64
        self.encoder = Encoder1Conv(ind, ind)
        self.encoder1_k = Encoder1Conv(ind, ind)
        self.encoder1_v = Encoder1Conv(ind, ind)

        self.fc_q = nn.Linear(ker * ker * ind, 1024)
        self.bn_q = nn.BatchNorm1d(1024)
        self.fc_k = nn.Linear(ker * ker * ind, 1024)
        self.bn_k = nn.BatchNorm1d(1024)

        self.feed = FeedForward(ind, ind * 4)

    def forward(self, feat):
        feature1_k = self.encoder1_k(feat)
        feature1_v = self.encoder1_v(feat)
        refer = self.encoder(feat)

        ### search
        q_un = F.unfold(refer, kernel_size=(ker, ker), stride=ker).transpose(1, 2)  # [N, Hr*Wr, C*k*k]
        k_un = F.unfold(feature1_k, kernel_size=(ker, ker), stride=ker).transpose(1, 2)
        v_un = F.unfold(feature1_v, kernel_size=(ker, ker), stride=ker).transpose(1, 2)

        q_un = self.bn_q(self.fc_q(q_un).transpose(1, 2)).transpose(1, 2)
        k_un = self.bn_k(self.fc_k(k_un).transpose(1, 2)).transpose(1, 2)

        k_un = k_un.permute(0, 2, 1)

        R_lv3 = torch.bmm(q_un, k_un)
        R_lv3 = R_lv3.div(q_un.size(2) ** 0.5)

        attn = F.softmax(R_lv3, dim=2)

        output = torch.bmm(attn, v_un).transpose(1, 2)

        output = F.fold(output, output_size=feat.size()[-2:], kernel_size=(ker, ker), stride=ker)

        output = self.feed(output) + feat

        return output


class Encoder1Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder1Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='zeros'),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class UNet_vit(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_vit, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.vit1 = SearchTransfer2(1024 // factor)
        self.up1 = Up(1024 + 512, 512 // factor, bilinear)

        self.vit2 = SearchTransfer2(512 // factor)
        self.up2 = Up(512 + 256, 256 // factor, bilinear)

        self.vit3 = SearchTransfer2(256 // factor)
        self.up3 = Up(256 + 128, 128 // factor, bilinear)

        self.vit4 = SearchTransfer2_tmp(128 // factor)
        self.up4 = Up(128 + 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        print("UNET_VIT")
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        xt4 = self.vit1(x4)
        x4 = torch.cat([x4, xt4], dim=1)
        x = self.up1(x5, x4)

        xt3 = self.vit2(x3)
        x3 = torch.cat([x3, xt3], dim=1)
        x = self.up2(x, x3)

        xt2 = self.vit3(x2)
        x2 = torch.cat([x2, xt2], dim=1)
        x = self.up3(x, x2)

        xt1 = self.vit4(x1)
        x1 = torch.cat([x1, xt1], dim=1)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
