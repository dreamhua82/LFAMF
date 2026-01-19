import torch
import torch.nn as nn
import math
from einops import rearrange


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(fn, fn, 3, padding=1)
        self.conv2 = nn.Conv2d(fn, fn, 3, padding=1)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


def make_layer(block, p, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(p))
    return nn.Sequential(*layers)


class D2Resblock(nn.Module):
    def __init__(self, channel):
        super(D2Resblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1),
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                dilation=(1, 1), bias=False)
    def __call__(self, x_init):
        x = self.conv(x_init)
        x = self.conv_2(x)
        return x + x_init


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.1, inplace=True)

        # Down sampling
        self.initConv = nn.Sequential(nn.Conv2d(in_dim, num_filters, 3, 1, 1),
                                      nn.LeakyReLU(0.1, True))
        self.down_1 = D2Resblock(self.num_filters)
        self.pool_1 = stride_conv_2d(self.num_filters, self.num_filters * 2, activation)
        self.down_2 = D2Resblock(self.num_filters * 2)
        self.pool_2 = stride_conv_2d(self.num_filters * 2, self.num_filters * 3, activation)

        # Bridge
        self.bridge_1 = D2Resblock(self.num_filters * 3)
        # self.bridge_2 = D3Resblock(self.num_filters * 3)

        # Up sampling
        self.trans_1 = conv_trans_block_2d(self.num_filters * 3, self.num_filters * 2, activation)
        self.up_1 = D2Resblock(self.num_filters * 2)
        self.trans_2 = conv_trans_block_2d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_2 = D2Resblock(self.num_filters * 1)

        # Output
        self.out_2D = nn.Conv2d(num_filters*2, 5, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Down sampling
        b, m, h, w = x.shape
        x = x.view(b * m, h, w).repeat(self.out_dim, 1, 1)
        x = rearrange(x, '(b n m) h w -> (b n) m h w', b=b, m=m)
        x = self.initConv(x)
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
        out = torch.cat((up_2, x), 1)
        out = self.out_2D(out)  # -> [1, 4, 128, 128, 128]
        out = rearrange(out, '(b n) c h w -> b n c h w', b=b)
        disp = out[:,:,0] # b 49 h w
        conf = out[:,:,1:] # b 49 4 h w
        return disp, conf


def conv_block_2d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
        activation)


def stride_conv_2d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=(2, 2), padding=1, bias=False),
        activation)


def conv_trans_block_2d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=(2, 2), padding=(1, 1),
                           output_padding=(1, 1), bias=False),
        activation)

if __name__ == '__main__':
    # Training settings
    import argparse

    parser = argparse.ArgumentParser(description="LF depth estimation: train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--step", type=int, default=250, help="Learning rate decay every n epochs")
    parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
    parser.add_argument("--patch_size", type=int, default=128, help="Training patch size")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
    parser.add_argument("--max_epoch", type=int, default=2000, help="maximum epoch for training")
    parser.add_argument("--num_cp", type=int, default=100, help="Number of epochs for saving checkpoint")
    parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")

    parser.add_argument("--dataset", type=str, default="HCI", help="Dataset for training")
    parser.add_argument("--dataset_path", type=str, default="./LFData/train_HCI_12LF_RGB.h5",
                        help="Dataset file for training")
    parser.add_argument("--angular_num", type=int, default=7, help="angular number of the light field ")
    parser.add_argument("--weight_smooth", type=float, default=0.1, help="weight for smooth loss ")
    parser.add_argument("--weight_conf", type=float, default=0.0, help="weight for confidence loss ")
    parser.add_argument("--loss_crop", type=int, default=8, help="crop the patch boundary when training")

    parser.add_argument("--std_thres", type=float, default=0.2)

    # parser.add_argument("--loss", type=str, default='MaskQuarterMinLoss')

    opt = parser.parse_args()
    opt.sub_lf_num = math.ceil(opt.angular_num / 2) ** 2
    net = UNet(4, 49, 64).cuda()
    x = torch.randn(1, 4, 128, 128).cuda()
    out = net(x)
    print(out[0].shape,out[1].shape)
    from thop import profile

    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of parameters: %.2fM' % (total / 1e6))
