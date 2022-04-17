CUDA_VISIBLE_DEVICES="0" \
python euvaluate_depth.py \
--data_path <yor_KITTI_path> \
--load_weights_folder <your_model_path>
--eval_mono