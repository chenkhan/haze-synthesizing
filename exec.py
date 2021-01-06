import os

os.system("python oldtrain.py --dataroot ./dataset/using/ --isTrain"
         " --learn_residual --resize_or_crop crop --display_freq 100 --print_freq 100 --display_port 8097"
         "   --niter 200 --no_html --batchSize 2 --save_epoch_freq 10 --lr_g 0.00005 --lr_d 0.00005")