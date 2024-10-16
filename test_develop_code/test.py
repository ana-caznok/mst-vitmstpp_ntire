import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR, SAMScore, SSIM_Loss
from hsi_dataset import TrainDataset, ValidDataset
from torch.utils.data import DataLoader

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--data_root', type=str, default='../dataset/')
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='./model_zoo/mst_plus_plus.pth')
parser.add_argument('--outf', type=str, default='./exp/mst_plus_plus/')
parser.add_argument('--no_crop', action='store_true', help="Do not crop the image")
parser.add_argument("--gpu_id", type=str, default='0')


opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if opt.no_crop:
    print("No cropping will be done.")
    crop = False
else:
    print("Cropping the image.")
    crop = True

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
print('opt.data_root', opt.data_root)
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = SAMScore()
criterion_ssim = SSIM_Loss()

if torch.cuda.is_available():
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_sam.cuda()
    criterion_ssim.cuda()

# Validate
print(f'{opt.data_root}/split_txt/valid_list.txt')
with open(f'{opt.data_root}/split_txt/valid_list.txt', 'r') as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()
var_name = 'cube'
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_sam = AverageMeter()
    losses_ssim = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            if method=='awan':   # To avoid out of memory, we crop the center region as input for AWAN.
                output = model(input[:, :, 118:-118, 118:-118])
                loss_mrae = criterion_mrae(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_rmse = criterion_rmse(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_psnr = criterion_psnr(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
            else:
                output = model(input)
                if crop ==True: 
                    loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                    loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                    loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                    loss_sam  = criterion_sam( output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                    loss_ssim = criterion_ssim(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                else: 
    
                    loss_mrae = criterion_mrae(output, target)
                    loss_rmse = criterion_rmse(output, target)
                    loss_psnr = criterion_psnr(output, target)
                    loss_sam  = criterion_sam( output, target)
                    loss_ssim = criterion_ssim(output, target)
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_sam.update( loss_sam.data)
        losses_ssim.update(loss_ssim.data)

        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        mat_name = hyper_list[i]
        mat_dir = os.path.join(opt.outf, mat_name)
        save_matv73(mat_dir, var_name, result)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_ssim.avg

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()
    mrae, rmse, psnr, sam, ssim = validate(val_loader, model)
    print(f'method:{method}, mrae:{mrae}, rmse:{rmse}, psnr:{psnr}, sam: {sam}, ssim: {ssim}')