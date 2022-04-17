CUDA_VISIBLE_DEVICES="0,1,2,3" \
python train.py \
--data_path /data/kitti_raw \
--log_dir ckpts  \
--model_name exp