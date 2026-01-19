import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from model_utility import *
class PVSNet(nn.Module):
    def __init__(self, angular_in, angular_out):
        super(PVSNet, self).__init__()
        channel = 48
        self.angRes = angular_in
        self.angRes_out = angular_out
        self.FeaExtract = InitFeaExtract(channel)
        self.D3Unet = UNet(channel, channel, channel, angular_in)
        self.Out = nn.Conv2d(channel, 1, 1, 1, 0, bias=False)
        self.Angular_UpSample = Upsample1(channel, angular_in, angular_out)
        self.Resup = Interpolation(angular_in, angular_out)

    def forward(self, x):
        x_mv = x.unsqueeze(2)
        b, n, c, h, w = x_mv.shape
        Bicubic_up = self.Resup(x_mv)

        buffer_mv_initial = self.FeaExtract(x_mv) # b n c h w

        buffer_mv = self.D3Unet(buffer_mv_initial.permute(0, 2, 1, 3, 4))

        HAR = self.Angular_UpSample(buffer_mv)

        out = self.Out(HAR.contiguous().view(b * self.angRes_out * self.angRes_out, -1, h, w))
        out = out.contiguous().view(b, -1, 1, h, w)
        out = FormOutput(out) + FormOutput(Bicubic_up)

        return out


class Upsample1(nn.Module):
    def __init__(self, channel, angular_in, angular_out):
        super(Upsample1, self).__init__()
        self.an = angular_in
        self.an_out = angular_out

        self.angconv = nn.Sequential(
                        nn.Conv3d(in_channels=channel*2, out_channels=channel*2, kernel_size = (3,3,3), stride = 1, padding=(1,1,1)),
                        nn.LeakyReLU(0.1, inplace=True))

        self.upsp = nn.Sequential(
            nn.ConvTranspose3d(channel*2, channel, kernel_size=3, stride=(1, 5, 5), padding=(1, 1, 1),
                               output_padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = rearrange(x, 'b (u v) c h w -> b c (h w) u v', u=self.an, v=self.an)
        x = self.angconv(x)
        out = self.upsp(x) # b c h*w u v
        out = rearrange(out, 'b c (h w) u v -> b (u v) c h w', h=h, w=w)

        return out

class Conv2d_refpad(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=3):
        super(Conv2d_refpad, self).__init__()
        pad = kernel // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernel, padding=0, bias=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channel, angular_in, angular_out):
        super(Upsample, self).__init__()
        self.an = angular_in
        self.angular_out = angular_out
        self.angconv = nn.Sequential(
                        nn.Conv2d(in_channels=channel*2, out_channels=channel*2, kernel_size = angular_in, stride = angular_in, padding=0),
                        nn.LeakyReLU(0.1, inplace=True))
        self.upsp = nn.Sequential(
            nn.Conv2d(channel*2, channel* angular_out * angular_out, kernel_size=1, padding=0),
            nn.PixelShuffle(angular_out),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b, n, c, h*w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b*h*w, c, self.an, self.an)
        up_in = self.angconv(x)
        out = self.upsp(up_in)
        out = rearrange(out, '(b h w) c u v -> b (u v) c h w', h=h, w=w)
        return out

class Interpolation(nn.Module):
    def __init__(self, angular_in, factor):
        super(Interpolation, self).__init__()
        self.an = angular_in
        self.an_out = factor#angular_in*factor
        self.factor = factor
    def forward(self, x_mv):
        b, n, c, h, w = x_mv.shape
        x = x_mv.contiguous().view(b, n, c, h*w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b*h*w, c, self.an, self.an)

        out = F.interpolate(x, size=(self.factor, self.factor), mode='bicubic', align_corners=False)#scale_factor=self.factor, mode='bicubic', align_corners=False)

        out = out.view(b,h*w,c,self.an_out*self.an_out)
        out = torch.transpose(out,1,3)
        out = out.contiguous().view(b, self.an_out*self.an_out, c, h, w)   #[N*81,c,h,w]
        return out

class D3Resblock(nn.Module):
    def __init__(self, channel):
        super(D3Resblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1),
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                                dilation=(1, 1, 1), bias=False)


    def __call__(self, x_init):
        x = self.conv(x_init)
        x = self.conv_2(x)
        return x + x_init


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters, an):
        super(UNet, self).__init__()
        self.an = an
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.1, inplace=True)

        # Down sampling
        self.down_1 = D3Resblock(self.in_dim)
        self.pool_1 = stride_conv_3d(self.num_filters, self.num_filters * 2, activation)
        self.down_2 = D3Resblock(self.num_filters * 2)
        self.pool_2 = stride_conv_3d(self.num_filters * 2, self.num_filters * 3, activation)

        # Bridge
        self.bridge_1 = D3Resblock(self.num_filters * 3)
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 3, self.num_filters * 2, activation)
        self.up_1 = D3Resblock(self.num_filters * 2)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_2 = D3Resblock(self.num_filters * 1)
        # Output
        self.out_2D = nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=3, stride=1, padding=1)
        self.alt = make_Altlayer(4, an, in_dim)
        self.out_2D1 = nn.Conv2d(num_filters * 3, num_filters * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        # Bridge
        bridge = self.bridge_1(pool_2)

        # Up sampling
        trans_1 = self.trans_1(bridge)
        addition_1 = trans_1 + down_2
        up_1 = self.up_1(addition_1)
        trans_2 = self.trans_2(up_1)
        addition_2 = trans_2 + down_1
        up_2 = self.up_2(addition_2)

        # Output
        out = torch.cat((up_2, x), 1).permute(0, 2, 1, 3, 4)
        b, n, c, h, w = out.shape  # z
        out = self.out_2D(out.contiguous().view(b * n, c, h, w))  # -> [1, 4, 128, 128, 128]
        alt = rearrange(x, 'b c (u v) h w -> (b u v) c h w', u=self.an, v=self.an)
        alt = self.alt(alt)
        out1 = torch.cat([out, alt], dim=1)
        out1 = self.out_2D1(out1).view(b, n, -1, h, w)
        return out1




def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
        activation)


def stride_conv_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
        activation)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1),
                           output_padding=(0, 1, 1), bias=False),
        activation)


class InitFeaExtract(nn.Module):
    def __init__(self, channel):
        super(InitFeaExtract, self).__init__()
        self.FEconv = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
          )
        # self.FEconv1 = nn.Sequential(
        #     nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        # #     nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
        # #     nn.LeakyReLU(0.1, inplace=True),
        # )

    def forward(self, x):
        b, n, r, h, w = x.shape  # b,4,1,h,w
        x = x.contiguous().view(b * n, -1, h, w)
        buffer = self.FEconv(x)
        # buffer = self.FEconv1(buffer) + buffer
        _, c, h, w = buffer.shape
        buffer = buffer.unsqueeze(1).contiguous().view(b, -1, c, h, w)  # .permute(0,2,1,3,4)

        return buffer


def FormOutput(x_sv):
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n + 1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk + 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


if __name__ == "__main__":
    net = PVSNet(2, 7)
    from thop import profile

    ##### get input index ######
    ind_all = np.arange(7 * 7).reshape(7, 7)
    delt = (7 - 1) // (2 - 1)
    ind_source = ind_all[0:7:delt, 0:7:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))
    ###
    input = torch.randn(1, 4, 32, 32)
    total = sum([param.nelement() for param in net.parameters()])
    # flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.4fM' % (total / 1e6))
    # print('   Number of FLOPs: %.4fG' % (flops / 1e9))
    out = net(input)
    print(out.shape)


