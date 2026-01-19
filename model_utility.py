
import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange
##########
def warping(disp, ind_source, ind_target, img_source, an):
    '''warping one source image/map to the target'''
    an2 = an*an
    N,h,w = img_source.shape
    ind_source = ind_source.type_as(disp)
    ind_target = ind_target.type_as(disp)
    ind_h_source = torch.floor(ind_source / an )
    ind_w_source = ind_source % an
    
    ind_h_target = torch.floor(ind_target / an)
    ind_w_target = ind_target % an

    # generate grid
    XX = torch.arange(0,w).view(1,1,w).expand(N,h,w).type_as(img_source) #[N,h,w]
    YY = torch.arange(0,h).view(1,h,1).expand(N,h,w).type_as(img_source)
    
    grid_w = XX + disp * (ind_w_target - ind_w_source)
    grid_h = YY + disp * (ind_h_target - ind_h_source)

    grid_w_norm = 2.0 * grid_w / (w-1) -1.0
    grid_h_norm = 2.0 * grid_h / (h-1) -1.0
            
    grid = torch.stack((grid_w_norm, grid_h_norm),dim=3) #[N,h,w,2]

    # inverse warp
    img_source = torch.unsqueeze(img_source,0)
    img_target = functional.grid_sample(img_source,grid, align_corners=False) # [N,1,h,w]
    img_target = torch.squeeze(img_target,1) #[N,h,w]
    return img_target

#############################

class AltFilter(nn.Module):
    def __init__(self, an, ch):
        super(AltFilter, self).__init__()

        self.spaconv = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.angconv = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.an = an
        self.an2 = an * an
        self.HepiConv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ch, ch, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(0.1, True)
        )
        self.VepiConv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ch, ch, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        N, c, h, w = x.shape  # [N*81,c,h,w]
        out = self.spaconv(x)
        out = rearrange(out, '(b u v) c h w -> (b h w) c u v', u=self.an, v=self.an)
        out = self.angconv(out)  # [N*h*w,c,9,9]
        out = rearrange(out, '(b h w) c u v -> (b u v) c h w ', h=h, w=w)
        epi = rearrange(x, '(b u v) c h w -> (b v w) c u h', u=self.an, v=self.an)
        epi = self.HepiConv(epi)
        epi = rearrange(epi, '(b v w) c u h -> (b u h) c v w', v=self.an, w=w)
        epi = self.VepiConv(epi)
        epi = rearrange(epi, '(b u h) c v w-> (b u v) c h w', u=self.an, h=h)
        return out + epi


class AltFilter1(nn.Module):
    def __init__(self, an, ch):
        super(AltFilter1, self).__init__()

        self.spaconv = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.angconv = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.an = an
        self.an2 = an * an
        self.HepiConv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.LeakyReLU(0.1, True),

        )
        self.VepiConv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        N, c, h, w = x.shape  # [N*81,c,h,w]
        out = self.spaconv(x)
        out = rearrange(out, '(b u v) c h w -> (b h w) c u v', u=self.an, v=self.an)
        out = self.angconv(out)  # [N*h*w,c,9,9]
        out = rearrange(out, '(b h w) c u v -> (b u v) c h w ', h=h, w=w)

        epi = rearrange(x, '(b u v) c h w -> (b v w) c u h', u=self.an, v=self.an)
        epi = self.HepiConv(epi)
        epi = rearrange(epi, '(b v w) c u h -> (b u h) c v w', v=self.an, w=w)
        epi = self.VepiConv(epi)
        epi = rearrange(epi, '(b u h) c v w-> (b u v) c h w', u=self.an, h=h)
        return out + epi

def make_Altlayer(layer_num, an, ch):
    layers = []
    for i in range( layer_num ):
        layers.append( AltFilter(an, ch) )
    return nn.Sequential(*layers)  


def make_Altlayer1(layer_num, an, ch):
    layers = []
    for i in range( layer_num ):
        layers.append( AltFilter1(an, ch) )
    return nn.Sequential(*layers)


class SpaConv(nn.Module):
    def __init__(self, an, feature_num):
        super(SpaConv, self).__init__()
        self.rcab = RCAB(feature_num,
                            reduction=16)
    def forward(self, x):
        out = self.rcab(x)
        return out


class AltFilter1(nn.Module):
    def __init__(self, an, feature_num):
        super(AltFilter1, self).__init__()
        self.an = an
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.spaconv = SpaConv(an, feature_num)
        self.angconv = RB(feature_num)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.an * self.an)
        out = self.spaconv(x)  # [N*an2,c,h,w]
        out = out.view(N, self.an * self.an, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = torch.reshape(out, (N * h * w, c, self.an, self.an))  # [N*h*w,c,an,an]
        out = self.angconv(out)  # [N*h*w,c,an,an]
        out = out.view(N, h * w, c, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = torch.reshape(out, (N * self.an * self.an, c, h, w))
        return out


class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x


class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels//reduction, channels//reduction, 3, 1, 2, 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels // reduction, channels // reduction, 3, 1, 2, 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels//reduction, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.last_conv = nn.Conv2d(2 * channels, channels, 1, bias=False)
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        t = self.sa(x)
        t = x * t
        y = x * y
        y = torch.cat([y, t], dim=1)
        y = self.last_conv(y)
        return y

class RCAB(nn.Module):
    def __init__(self, in_channels, reduction):
        super(RCAB, self).__init__()
        out_channels = in_channels
        self.conv3_1_A = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_1_B = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.relu_1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_2_A = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_2_B = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.adafm = nn.Conv2d(out_channels, out_channels, 3, padding=3//2, groups=out_channels)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, input):
        x = self.conv3_1_A(input) + self.conv3_1_B(input)
        x = self.relu_1(x)
        x = self.conv3_2_A(x) + self.conv3_2_B(x)
        x = self.adafm(x)+x
        x = self.ca(x)
        return input + x