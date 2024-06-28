#python train.py --method hrnet  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/hrnet-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0
#python train.py --method hscnn_plus  --batch_size 4 --end_epoch 500 --init_lr 4e-4 --outf ./exp/hscnn_plus-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0
#python train.py --method mst_plus_plus  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/mst_plus_plus-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True
#python train.py --method vitmstpp  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/vitmstpp-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True
python train.py --method vitmstpp_pad  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/vitmstpp_pad-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True
python train.py --method restormer  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/restormer-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True
python train.py --method hscnn_plus  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/hscnn_plus-exp4/ --data_root ../dataset/  --patch_size 256 --stride 64  --gpu_id 0 --wandb True

python train.py --method vitmstpp_pad  --batch_size 4 --end_epoch 100 --init_lr 4e-4 --outf ./exp/vitmstpp_pad-v2-exp4/ --data_root ../dataset/  --patch_size 482 --stride 8  --gpu_id 0 --wandb True