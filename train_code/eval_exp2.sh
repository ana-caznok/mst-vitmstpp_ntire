








python test.py --data_root /mnt/datassd/icasp/data/NTIRE/  --method mst_plus_plus --pretrained_model_path /mnt/datassd/mst_toolbox/MST-plus-plus-2/train_code/exp/mst_plus_plus-exp2/2024_06_03_17_50_06/net_54epoch.pth --outf ./exp/mst_plus_plus_exp2/  --gpu_id 0

python test.py --data_root /mnt/datassd/icasp/data/NTIRE/  --method restormer --pretrained_model_path /mnt/datassd/mst_toolbox/MST-plus-plus-2/train_code/exp/restormer-exp2/2024_06_01_17_53_02/net_625epoch.pth --outf ./exp/restormer_exp2/  --gpu_id 0 

python test.py --data_root /mnt/datassd/icasp/data/NTIRE/  --method mst_plus_plus --pretrained_model_path /mnt/datassd/mst_toolbox/MST-plus-plus-2/train_code/exp/mstpp_exp4/net_194epoch.pth --outf ./exp/mstpp_exp4/  --gpu_id 0

python test.py --data_root /mnt/datassd/icasp/data/NTIRE/  --method vitmstpp --pretrained_model_path /mnt/datassd/mst_toolbox/MST-plus-plus-2/train_code/exp/vitmstpp_exp4/net_71epoch.pth --outf ./exp/vitmstpp_exp4/  --gpu_id 0

python test.py --data_root /mnt/datassd/icasp/data/NTIRE/  --method vitmstpp_pad --pretrained_model_path /mnt/datassd/mst_toolbox/MST-plus-plus-2/train_code/exp/vitmstpp-pad_exp4/net_58epoch.pth --outf ./exp/vitmstpp-pad_exp4/  --gpu_id 0