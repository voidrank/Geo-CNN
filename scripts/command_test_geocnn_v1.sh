#/bin/bash
python3 train/test.py --gpu $1 --num_point 1024 --model frustum_geocnn_v1 --model_path train/log_geocnn_v1/model.ckpt --output train/detection_geocnn_results_v1 --data_path kitti/frustum_carpedcyc_val_rgb_detection.pickle --from_rgb_detection --idx_path kitti/image_sets/val.txt --from_rgb_detection
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_geocnn_reseuls_v1/data
