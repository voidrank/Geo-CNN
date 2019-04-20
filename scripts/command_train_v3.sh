#/bin/bash
export CUDA_VISIBLE_DIVICES=$1; python3 train/train.py --gpu 0 --model frustum_pointnets_v3 --log_dir train/log_v3 --num_point 1024 --max_epoch 201 --batch_size 16 --decay_step 800000 --decay_rate 0.5
