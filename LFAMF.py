import torch
import torch.nn as nn
import torch.nn.functional as functional
from model_utility import *
from Unet import UNet
from PVSNet import PVSNet as ASRnet
from einops import rearrange
import numpy as np

class Net(nn.Module):
    def __init__(self, opt):

        super(Net, self).__init__()
        channel = 48
        an2 = opt.angular_out * opt.angular_out
        # disparity
        self.ASR = ASRnet(opt.angular_in, opt.angular_out)
        self.disp_estimator = UNet(opt.num_source, an2, channel)
        # LF
        self.lf_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=opt.num_source + 1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

        )
        self.lf_altblock = make_Altlayer1(3, an=opt.angular_out, ch=64)

        if opt.angular_out == 8:
            self.lf_res_conv = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),
                # 64-->16
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),
                # 16-->4
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                # 4-->1
            )

        if opt.angular_out == 7:
            self.lf_res_conv = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),
                # 49-->12
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=49, kernel_size=(3, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1)),
                # 12-->4

            )
        self.softmax_d2 = nn.Softmax(dim=2)

    def forward(self, img_source, cfg):
        ind_all = np.arange(cfg.angular_out * cfg.angular_out).reshape(cfg.angular_out, cfg.angular_out)
        delt = (cfg.angular_out - 1) // (cfg.angular_in - 1)
        ind_source = ind_all[0:cfg.angular_out:delt, 0:cfg.angular_out:delt]
        ind_source = torch.from_numpy(ind_source.reshape(-1))
        an = cfg.angular_out
        an2 = cfg.angular_out * cfg.angular_out

        img_source = rearrange(img_source, 'b 1 (u h) (v w) -> b (u v) h w', u=cfg.angular_in, v=cfg.angular_in)

        # ind_source
        N, num_source, h, w = img_source.shape  # [N,num_source,h,w]
        #################### disparity estimation ##############################
        disp_target, confD = self.disp_estimator(img_source)  # [N,an2,h,w]

        #################### intermediate LF ##############################
        warp_img_input = img_source.view(N * num_source, 1, h, w).repeat(an2, 1, 1, 1)  # [N*an2*4,1,h,w]

        grid = []
        for k_t in range(0, an2):
            for k_s in range(0, num_source):
                ind_s = ind_source[k_s].type_as(img_source)
                ind_t = torch.arange(an2)[k_t].type_as(img_source)
                ind_s_h = torch.floor(ind_s / an)
                ind_s_w = ind_s % an
                ind_t_h = torch.floor(ind_t / an)
                ind_t_w = ind_t % an
                disp = disp_target[:, k_t, :, :]

                XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(img_source)  # [N,h,w]
                YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(img_source)
                grid_w_t = XX + disp * (ind_t_w - ind_s_w)
                grid_h_t = YY + disp * (ind_t_h - ind_s_h)
                grid_w_t_norm = 2.0 * grid_w_t / (w - 1) - 1.0
                grid_h_t_norm = 2.0 * grid_h_t / (h - 1) - 1.0
                grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm), dim=3)  # [N,h,w,2]
                grid.append(grid_t)
        grid = torch.cat(grid, 0)  # [N*an2*4,h,w,2]

        warped_img = functional.grid_sample(warp_img_input, grid, align_corners=True).view(N, an2, num_source, h, w)
        asr_feat = self.ASR(img_source) # SAI

        asr_feat = rearrange(asr_feat, 'b c (u h) (v w) -> b (u v) c h w', u=an, v=an)
        fusion_img = torch.cat([warped_img, asr_feat],dim=2)
        conf = self.softmax_d2(confD)
        fusion = torch.sum(warped_img*conf, dim=2)
        ################# refine LF ###########################
        feat = self.lf_conv0(fusion_img.view(N * an2, num_source + 1, h, w))  # [N*an2,64,h,w]
        feat = self.lf_altblock(feat)  # [N*an2,64,h,w]
        feat = torch.transpose(feat.view(N, an2, 64, h, w), 1, 2)  # [N,64,an2,h,w]
        res = self.lf_res_conv(feat)  # [N,an2,6,h,w]
        asr_res = res[:,:,0,:,:]
        asr_res = asr_res.unsqueeze(2) + asr_feat
        warp_res = res[:, :, 1, :, :]
        warp_res = warp_res.unsqueeze(2) + fusion.unsqueeze(2)
        asr_warp = torch.cat([warp_res, asr_res], dim=2)
        conf_fusion = res[:, :, 2:, :, :]
        conf_fusion = self.softmax_d2(conf_fusion) # b 49 2 h w
        lf = torch.sum(asr_warp*conf_fusion, dim=2)
        return asr_feat.squeeze(2), disp_target, lf, fusion

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Light Field Hybrid SR")

    # training settings
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--step", type=int, default=500, help="Learning rate decay every n epochs")
    parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
    parser.add_argument("--patch_size", type=int, default=96, help="Training patch size")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
    parser.add_argument("--num_cp", type=int, default=25, help="Number of epochs for saving checkpoint")
    parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")

    parser.add_argument("--smooth", type=float, default=0.001, help="smooth loss")
    parser.add_argument("--epi", type=float, default=1.0, help="epi loss")

    parser.add_argument("--dataset", type=str, default="HCI", help="Dataset for training")
    parser.add_argument("--dataset_path", type=str, default="./LFData/train_HCI.h5",
                        help="H5 file containing the dataset for training")
    # model
    parser.add_argument("--layer_num", type=int, default=4, help="layer_num of SAS")
    parser.add_argument("--angular_out", type=int, default=7, help="angular number of the dense light field")
    parser.add_argument("--angular_in", type=int, default=2,
                        help="angular number of the sparse light field [AngIn x AngIn]")

    opt = parser.parse_args()
    opt.num_source = opt.angular_in * opt.angular_in

    model = Net(opt)
    input = torch.randn(2, 1, 64, 64)  # b 4 h w    /  b 49 h w
    ind_source = torch.tensor([0, 6, 42, 48])
    out = model(input, opt)
    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
    total = sum([param.nelement() for param in model.parameters()])
    # flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.4fM' % (total / 1e6))
    # print('   Number of FLOPs: %.4fG' % (flops / 1e9))