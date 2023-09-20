cd src
# train
python main.py tracking --exp_id kitti_half --dataset kitti_tracking --dataset_version train_half --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0,1 --batch_size 16 --load_model ../models/nuScenes_3Ddetection_e140.pth
# test
python test.py tracking --exp_id kitti_half --dataset kitti_tracking --dataset_version val_half --pre_hm --track_thresh 0.4 --resume


# dep ratio head train
python main.py ddd,dep_ratio --exp_id kitti_half --dataset kitti_tracking --dataset_version train_MiD --batch_size 16 --gpus 0,1


python test.py ddd,tracking,dep_ratio --exp_id kitti_half --dataset kitti_tracking --dataset_version val_MiD --gpus 0 --load_model todo_take_above_model


