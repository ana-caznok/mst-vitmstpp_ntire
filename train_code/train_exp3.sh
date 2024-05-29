python train.py --method vitmstpp  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/vitmstpp-exp3/ --data_root ../dataset/  --patch_size 256 --stride 8  --gpu_id 0
python train.py --method vitmstpp_pad  --batch_size 2 --end_epoch 500 --init_lr 4e-4 --outf ./exp/vitmstpp_pad-exp3/ --data_root ../dataset/  --patch_size 256 --stride 8  --gpu_id 0
