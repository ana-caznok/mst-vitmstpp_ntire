import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime
import wandb
import time 


parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument("--wandb", type=bool, default=False, help='log files in wandb')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = len(train_data)//opt.batch_size #era 1000
total_iteration = per_epoch_iteration*opt.end_epoch
checkpoint_save = per_epoch_iteration//2

print('ITERATIONS PER EPOCH: ', per_epoch_iteration)
print('TOTAL ITERATION: ', total_iteration)

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(method, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

if opt.wandb == True: 
    f_configurations = {'method': opt.method, 
                        'batch_size': opt.batch_size, 
                        'patch_size':opt.patch_size, 
                        'stride':opt.stride, 
                        'epoch_end': opt.end_epoch, 
                        'lr': opt.init_lr,
                        'train_imgs': train_data.img_num, 
                        'valid_imgs:':len(val_data)
                        }
    run = wandb.init(project="hyperskin-challenge",
                                reinit=True,
                                config=f_configurations,
                                notes="Running experiment",
                                entity="rainbow-ai",
                                name=method +'-ntire-' + str(opt.patch_size)+'-cw')

    for k in f_configurations.keys():
        wandb.config[k] = f_configurations[k]

# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    cudnn.benchmark = True
    iteration = 0
    ep=0
    record_mrae_loss = checkpoint_save
    start = time.time()

    while iteration<total_iteration: #vai até o número total de iterações 
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        print('len train loader: ', len(train_loader))
        ep=ep+1
         
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_mrae(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration+1

            if iteration % 100 == 0:
                elaps_time = time.time() - start
                predicted_time = (total_iteration/iteration)*elaps_time
                time_left = predicted_time - elaps_time
                print('[epoch:%d/%d][iter:%d/%d],lr=%.9f,train_losses.avg=%.9f, time left=%d,  time passed=%d, predicted time=%d'
                      % (ep,opt.end_epoch, iteration, total_iteration, lr, losses.avg, time_left, elaps_time, predicted_time))
                if opt.wandb == True: 
                    wandb.log({"avg_losses": losses.avg}) #new new
                
            if iteration % checkpoint_save ==0: #1000 == 0: a validação está acontecendo mais de uma vez por época! atenção 
                print('iteration>total_iteration: ', iteration>total_iteration)
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')

                if opt.wandb == True: 
                    wandb.log({"mrae_loss": mrae_loss}) #new
                    wandb.log({"rmse_loss": rmse_loss}) #new
                    wandb.log({"psnr_loss": psnr_loss}) #new

                # Save model. O salvamento do modelo também acontece 1x/época 
                if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // checkpoint_save), iteration, model, optimizer) #era 1000
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss
                        wandb.log({"best_val_loss": record_mrae_loss}) #new
                # print loss
                
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//per_epoch_iteration, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss)) #era 1000
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//per_epoch_iteration, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss)) #era 1000
                #if iteration>total_iteration: 
                #    break
            
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)

    if opt.wandb == True: 
        wandb.log({"mrae_val_loss": losses_mrae.avg}) #new
        wandb.log({"rmse_val_loss": losses_rmse.avg}) #new
        wandb.log({"psnr_val_loss": losses_psnr.avg}) #new
        
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

if __name__ == '__main__':
    main()
    print(torch.__version__)