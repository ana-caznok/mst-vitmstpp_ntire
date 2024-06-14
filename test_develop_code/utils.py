import hdf5storage
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from fvcore.nn import FlopCountAnalysis
from HyperSkinUtils.hyper_utils import sam_fn
from typing import List
from torchmetrics.image import StructuralSimilarityIndexMeasure

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.view(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def my_summary(test_model, H = 256, W = 256, C = 31, N = 1):
    model = test_model.cuda()
    print(model)
    inputs = torch.randn((N, C, H, W)).cuda()
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')


class SAMScore(nn.Module):
    '''
    Returns the score value from Challenge owners sam_fn
    '''
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert len(target.shape) == 4 and len(pred.shape) == 4, "SAMScore accepts a 4D batch as an input"

        sam_scores: List[torch.Tensor] = []
        for p, t in zip(pred, target):
            sam_scores.append(sam_fn(p, t)[0])
        
        return torch.stack(sam_scores).mean()
    
    def reset(self):
        pass

class SSIM_Loss(nn.Module):
    def __init__(self, enable_ssim=True):
        super().__init__()
        
        self.enable_ssim = enable_ssim
        
        assert int(enable_ssim) > 0, "Please enable some loss"
        
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, pred, target):
        loss = 0
        pred += 1e-5
        target += 1e-5
        
        if self.enable_ssim:
            loss = self.ssim(pred, target)
        
        return loss
    
    def reset(self):
        
        self.ssim.reset()

    def __str__(self):
        return f"MRAESSIMSAMLoss - SSIM: {self.enable_ssim}"
