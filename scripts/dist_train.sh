CUDA_VISIBLE_DEVICES="0,1,3,4" \
python -m torch.distributed.launch --nproc_per_node 2 --master_port 29502 train_ddp.py \
--log_dir ckpts \
--data_path /data/kitti_raw \
--model_name exp \
--learning_rate 1e-4 \
--batch_size 16 \
--freeze_teacher_epoch 15 \
--height 192 --width 640