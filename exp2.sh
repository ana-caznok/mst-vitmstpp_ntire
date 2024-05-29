python train.py --method mst_plus_plus  --batch_size 2 --end_epoch 300 --init_lr 4e-4 --outf ./exp/mst_plus_plus/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
python train.py --method vitmstpp  --batch_size 2 --end_epoch 300 --init_lr 4e-4 --outf ./exp/vitmstpp/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
python train.py --method edsr  --batch_size 2 --end_epoch 300 --init_lr 4e-4 --outf ./exp/restormer/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
python train.py --method hdnet  --batch_size 2 --end_epoch 300 --init_lr 4e-4 --outf ./exp/hdnet/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
python train.py --method hrnet  --batch_size 2 --end_epoch 300 --init_lr 4e-4 --outf ./exp/hrnet/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

