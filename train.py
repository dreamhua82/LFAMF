import time
import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils1.utils import *
from LFAMF import Net
from tqdm import tqdm
import math
from einops import rearrange
# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument("--angRes_in", type=int, default=2, help="input angular resolution")
    parser.add_argument('--trainset_dir', type=str, default='./Data/TrainingData_HCI_2x2_ASR_7x7/')
    parser.add_argument('--testset_dir', type=str, default='./Data/TestData_HCI_2x2_ASR_7x7/')
    parser.add_argument('--model_name', type=str, default='HCI_2x2-7x7')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=70, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches for validation")
    parser.add_argument("--patchsize", type=int, default=128, help="LFs are cropped into patches for validation")
    parser.add_argument("--stride", type=int, default=64, help="LFs are cropped into patches for validation")
    
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='./log/HCI_2x2-7x7.pth')

    parser.add_argument("--smooth", type=float, default=0.001, help="smooth loss")
    parser.add_argument("--epi", type=float, default=1.0, help="epi loss")
    # model
    parser.add_argument("--layer_num", type=int, default=4, help="layer_num of SAS")
    parser.add_argument("--angular_out", type=int, default=7, help="angular number of the dense light field")
    parser.add_argument("--angular_in", type=int, default=2,
                        help="angular number of the sparse light field [AngIn x AngIn]")

    return parser.parse_args()


def reconstruction_loss(X, Y):
    # L1 Charbonnier loss
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    loss = torch.sum(error) / torch.numel(error)
    return loss

def gradient(pred):
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def smooth_loss(pred_map):
    # [N,an2,h,w]
    loss = 0
    weight = 1.
    dx, dy = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
    return loss

def epi_loss(pred, label):
    # epi loss
    def lf2epi(lf):
        N, an2, h, w = lf.shape
        an = int(math.sqrt(an2))
        epi_h = lf.view(N, an, an, h, w).permute(0, 1, 3, 2, 4).contiguous().view(-1, 1, an, w)
        epi_v = lf.view(N, an, an, h, w).permute(0, 2, 4, 1, 3).contiguous().view(-1, 1, an, h)
        return epi_h, epi_v

    epi_h_pred, epi_v_pred = lf2epi(pred)
    dx_h_pred, dy_h_pred = gradient(epi_h_pred)
    dx_v_pred, dy_v_pred = gradient(epi_v_pred)

    epi_h_label, epi_v_label = lf2epi(label)
    dx_h_label, dy_h_label = gradient(epi_h_label)
    dx_v_label, dy_v_label = gradient(epi_v_label)

    return reconstruction_loss(dx_h_pred, dx_h_label) + reconstruction_loss(dy_h_pred,
                                                                            dy_h_label) + reconstruction_loss(dx_v_pred,
                                                                                                              dx_v_label) + reconstruction_loss(
        dy_v_pred, dy_v_label)




def train(cfg, train_loader, test_Names, test_loaders):
    ##### get input index ######

    net = Net(cfg)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    if cfg.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3, 4])

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            epoch_state = model["epoch"]
            optimizer.load_state_dict(model['optimizer'])
            scheduler.load_state_dict(model['scheduler'])
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    scheduler._step_count = epoch_state
    loss_epoch = []
    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            # forward pass
            label = rearrange(label, 'b 1 (u h) (v w) -> b (u v) h w', u=cfg.angular_out,v=cfg.angular_out)
            asr, disp, pred_lf, fusion = net(data, cfg)
            pred_lf = rearrange(pred_lf, 'b 1 (u h) (v w) -> b (u v) h w', u=cfg.angular_out,v=cfg.angular_out)

            loss = 5 * reconstruction_loss(pred_lf, label) + cfg.smooth * smooth_loss(disp)
            loss += reconstruction_loss(fusion, label)
            loss += reconstruction_loss(asr, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f, lr---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), scheduler.get_last_lr()[0]))
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, save_path='./log/', filename=cfg.model_name + '_epoch_{}.pth'.format(idx_epoch + 1))

            loss_epoch = []

        ''' evaluation '''
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []
            for index, test_name in enumerate(test_Names):
                test_loader = test_loaders[index]
                psnr_epoch_test, ssim_epoch_test = valid(test_loader, net)
                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                print(time.ctime()[4:-5] + ' Dataset----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
                pass
            pass
        print(time.ctime()[4:-5] + 'average psrn:%f, ssim:%f,' % (
        float(np.array(psnr_testset).mean()), float(np.array(ssim_testset).mean())))
        scheduler.step()
        pass


def valid(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze().to(cfg.device)
        if cfg.crop == False:
            with torch.no_grad():
                outLF = net(data.unsqueeze(0).unsqueeze(0).to(cfg.device))
                outLF = outLF.squeeze()
        else:
            uh, vw = data.shape
            h0, w0 = uh // cfg.angRes_in, vw // cfg.angRes_in
            subLFin = LFdivide(data, cfg.angRes_in, cfg.patchsize, cfg.patchsize // 2)  # numU, numV, h*angRes, w*angRes
            numU, numV, H, W = subLFin.shape
            subLFout = torch.zeros(numU, numV, cfg.angRes_out * cfg.patchsize, cfg.angRes_out * cfg.patchsize)

            for u in range(numU):
                for v in range(numV):
                    tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        _,_,out,_ = net(tmp.to(cfg.device),cfg)
                        subLFout[u, v, :, :] = out.squeeze()
            outLF = LFintegrate(subLFout, cfg.angRes_out, cfg.patchsize, cfg.stride, h0, w0)
        psnr, ssim = cal_metrics_RE(label, outLF, cfg.angRes_in, cfg.angRes_out)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    return psnr_epoch_test, ssim_epoch_test


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=12, batch_size=cfg.batch_size, shuffle=True)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, train_loader, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    cfg.num_source = cfg.angular_in * cfg.angular_in
    main(cfg)
