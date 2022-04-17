import argparse
import os
import numpy as np
import cv2
from pathlib import Path
import pathlib
from tqdm import tqdm
from array import array
from PLYUtil import PLYUtils
import open3d as o3d
import pysnooper

def build_single_frame_point_cloud(dataset_path, scale=100.0):
    plyutils = PLYUtils(max_d=130)
    # dataset_folder = Path(dataset_path)

    K = np.fromfile(args.intrinsic, dtype=float, sep="\n ").reshape((3, 3))
    image_files = sorted(Path(args.image).glob("*.jpg"))
    depth_files = sorted(Path(args.depth).glob("*.png"))
    # 这里使用pose的真值
    poses = np.eye(4)

    # for i in tqdm(range(0,1)):
    for i in tqdm(range(len(image_files))):
        image_file = str(image_files[i])
        depth_file = str(depth_files[i])

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)

        height, weight, _ = rgb.shape
        depth = cv2.resize(depth, (weight, height))
        current_points_3D = plyutils.add_depthmap(depth, rgb, scale=scale, intrinsics=K, extrinsics=poses)
        print("point_3d ", current_points_3D.shape)

        save_ply_name = os.path.basename(os.path.splitext(image_files[i])[0]) + ".ply"
        save_ply_path = Path(args.output)
        # print(str(save_ply_path))
        # print(save_ply_path, save_ply_name)
        if not os.path.exists(str(save_ply_path)):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(save_ply_path)
        PLYUtils.save(str(save_ply_path / save_ply_name), current_points_3D)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", help="color image segmentation path")
    parser.add_argument("--depth", "-d", help="depth map path")
    parser.add_argument("--intrinsic", "-in", help="camera intrinsic txt path")
    parser.add_argument("--output", help="output point cloud path")

    args = parser.parse_args()

    build_single_frame_point_cloud(args)
