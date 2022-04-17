import argparse
import os
import numpy as np
import cv2
from pathlib import Path
import pathlib
from tqdm import tqdm
from array import array
from PLYUtil import PLYUtils

# def build_concat_cloud(dataset_path, output_path, scale=100.0):
def build_concat_cloud(args, scale=100.0):
    plyutils = PLYUtils(max_d=50)

    K = np.fromfile(args.intrinsic, dtype=float, sep="\n ").reshape((3, 3))
    image_files = sorted(Path(args.image).glob("*.jpg"))
    depth_files = sorted(Path(args.depth).glob("*.png"))
    seg_files = sorted(Path(args.segmentation).glob("*.png"))
    # 这里使用pose的真值
    poses = np.loadtxt(args.pose).reshape((-1, 3, 4))

    # for i in tqdm(range(20)):
    for i in tqdm(range(len(image_files))):
        image_file = str(image_files[i])
        depth_file = str(depth_files[i])
        seg_file = str(seg_files[i])

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)
        seg = cv2.imread(seg_file, 0)

        pose = np.concatenate((poses[i], np.array([[0, 0, 0, 1]])), axis=0)

        height, weight, _ = rgb.shape
        depth = cv2.resize(depth, (weight, height))

        assert depth.shape[:2] == rgb.shape[:2], "depthmap and rgb image size mismatch"
        # plyutils.add_depthmap(depth, rgb, scale, seg=seg, intrinsics=K, extrinsics=pose)
        plyutils.add_depthmap(depth, rgb, scale, seg=seg, intrinsics=K, extrinsics=pose)

    plyutils.write_point_cloud(args.output)

    # print(args.vis)
    if args.vis:
        PLYUtils.draw_pointcloud(args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", help="color image segmentation path")
    parser.add_argument("--depth", "-d", help="depth map path")
    parser.add_argument("--segmentation", "-seg", help="segmentation png path")
    parser.add_argument("--pose", help="pose txt path")
    parser.add_argument("--intrinsic", "-in", help="camera intrinsic txt path")
    parser.add_argument("--output", help="output point cloud path")
    parser.add_argument("--vis", help="if vis point cloud output")

    args = parser.parse_args()

    build_concat_cloud(args)
