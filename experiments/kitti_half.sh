cd src
# train
#python main.py tracking --exp_id kitti_half --dataset kitti_tracking --dataset_version train_half --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0,1 --batch_size 16 --load_model ../models/nuScenes_3Ddetection_e140.pth

# train with depth, dep_ratio heads on MiD data split.
python main.py tracking,ddd --exp_id kitti_half_MiD_train --dataset kitti_tracking --dataset_version train_MiD --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0,1 --batch_size 16 --load_model ../models/nuScenes_3Ddetection_e140.pth

# test
#python test.py tracking --exp_id kitti_half --dataset kitti_tracking --dataset_version val_half --pre_hm --track_thresh 0.4 --resume

# test with depth, dep_ratio heads on MiD data split.
python test.py tracking,ddd --exp_id kitti_half_MiD_test --dataset kitti_tracking --dataset_version val_MiD --pre_hm --track_thresh 0.4 --load_model todo_add_path_to_pth_file