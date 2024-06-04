# python train.py --method hrnet  --batch_size 2 --end_epoch 100 --init_lr 4e-4 --outf ./exp/hrnet-exp2/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
# python train.py --method hscnn_plus  --batch_size 2 --end_epoch 100 --init_lr 4e-4 --outf ./exp/hscnn_plus-exp2/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
python train.py --method mst_plus_plus  --batch_size 1 --end_epoch 30 --init_lr 4e-4 --outf ./exp/mst_plus_plus-exp2/ --data_root ../dataset/  --patch_size 128 --stride 64  --gpu_id 0 --wandb True
python train.py --method vitmstpp  --batch_size 1 --end_epoch 30 --init_lr 4e-4 --outf ./exp/vitmstpp-exp2/ --data_root ../dataset/  --patch_size 128 --stride 64  --gpu_id 0 --wandb True
python train.py --method vitmstpp_pad  --batch_size 1 --end_epoch 30 --init_lr 4e-4 --outf ./exp/vitmstpp_pad-exp2/ --data_root ../dataset/  --patch_size 128 --stride 64  --gpu_id 0 --wandb True
python train.py --method restormer  --batch_size 1 --end_epoch 30 --init_lr 4e-4 --outf ./exp/restormer-exp2/ --data_root ../dataset/  --patch_size 128 --stride 64  --gpu_id 0 --wandb True
python train.py --method hscnn_plus  --batch_size 1 --end_epoch 30 --init_lr 4e-4 --outf ./exp/hscnn_plus-exp2/ --data_root ../dataset/  --patch_size 128 --stride 64  --gpu_id 0 --wandb True