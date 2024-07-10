#python train.py --method hrnet  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/hrnet-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0
#python train.py --method hscnn_plus  --batch_size 4 --end_epoch 500 --init_lr 4e-4 --outf ./exp/hscnn_plus-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0
#python train.py --method mst_plus_plus  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/mst_plus_plus-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True
#python train.py --method vitmstpp  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/vitmstpp-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True
#python train.py --method vitmstpp_pad  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/vitmstpp_pad-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True
#python train.py --method restormer  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/restormer-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True
#python train.py --method hscnn_plus  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/hscnn_plus-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True
#python train_cw.py --method vitmstpp_pad  --batch_size 2 --end_epoch 100 --init_lr 4e-4 --outf ./exp/vitmstpp_pad-exp4_cw/ --data_root ../dataset/  --patch_size 482 --stride 15  --gpu_id 0



python train_cw.py --method vitmstpp_pad  --pretrained_model_path /mnt/datassd/mst_toolbox/mst-vitmstpp_ntire/train_code/exp/vitmstpp_pad-exp4_cw/2024_07_09_12_01_58/net_151epoch.pth --batch_size 4 --end_epoch 25 --init_lr 4e-4 --outf ./exp/vitmstpp_pad-exp4_cw/ --data_root ../dataset/  --patch_size 482 --stride 15  --gpu_id 0

python train_cw.py --method vitmstpp_pad  --pretrained_model_path /mnt/datassd/mst_toolbox/mst-vitmstpp_ntire/train_code/exp/vitmstpp_pad-exp4/2024_06_19_10_40_25/net_93epoch.pth --batch_size 4 --end_epoch 50 --init_lr 4e-4 --outf ./exp/vitmstpp_pad-exp4_cw/ --data_root ../dataset/  --patch_size 482 --stride 15  --gpu_id 0
