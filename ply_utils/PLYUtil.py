import os
import numpy as np
import cv2
from pathlib import Path
import pathlib
from tqdm import tqdm
from array import array
import open3d as o3d
import pysnooper
from loguru import logger
import matplotlib.pyplot as plt

PALETTE = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
             [180, 165, 180], [90, 120, 150], [
               102, 102, 156], [128, 64, 255],
             [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
             [230, 150, 140], [128, 64, 128], [
               110, 110, 110], [244, 35, 232],
             [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
             [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
             [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
             [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
             [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
             [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
             [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
             [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
             [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
             [119, 11, 32], [150, 0, 255], [
               0, 60, 100], [0, 0, 142], [0, 0, 90],
             [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70],
             [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]]

CLASSES = ('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier',
           'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area',
           'Rail Track', 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel',
           'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',
           'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation',
           'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
           'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
           'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
           'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat',
           'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer',
           'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled')

# PALETTE = [[210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
#              [250, 170, 30], [192, 192, 192], [220, 220, 0]]
seg_list = [2, 8, 13, 15, 23, 24, 29]
# seg_list = [2, 9, 23, 24]
show_seg_list = [2, 8, 9, 23, 24]

class PLYUtils:
    def __init__(self, min_d=3, max_d=20, batch_size=1, roi=[160, 360, 20, 1200], dropout=0):
        super(PLYUtils, self).__init__()
        self.min_d = min_d
        self.max_d = max_d
        self.roi = roi
        self.dropout = dropout

        # 第一列是xyz的最小值，第二列是xyz的最大值
        self.voxel_bound = np.zeros((3, 2))
        self.voxel_bound[:, 0] = np.inf

        self.data = array('f')

    def write_point_cloud(self, ply_filename):
        length = len(self.data) // 6

        header = "ply\n" \
                 "format binary_little_endian 1.0\n" \
                 f"element vertex {length}\n" \
                 f"property float x\n" \
                 f"property float y\n" \
                 f"property float z\n" \
                 f"property float red\n" \
                 f"property float green\n" \
                 f"property float blue\n" \
                 f"end_header\n"

        with open(ply_filename, "wb") as file:
            file.write(header.encode(encoding="ascii"))
            self.data.tofile(file)

    '''
    用于函数外部保存点云数据
    '''
    @staticmethod
    def save(ply_filename, points):
        formatted_points = []
        for point in points:
            formatted_points.append(
                "%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], int(point[3]), int(point[4]), int(point[5])))

        out_file = open(ply_filename, "w")
        out_file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(formatted_points)))
        out_file.close()

    def backproject(self, depth, scale, intrinsics):
        u = range(0, depth.shape[1])
        v = range(0, depth.shape[0])

        u, v = np.meshgrid(u, v)
        u, v= u.astype(float), v.astype(float)

        Z = depth.astype(float) / scale
        X = (u - intrinsics[0, 2]) * Z / intrinsics[0, 0]
        Y = (v - intrinsics[1, 2]) * Z / intrinsics[1, 1]

        X = np.ravel(X)
        Y = np.ravel(Y)
        Z = np.ravel(Z)

        return np.vstack((X, Y, Z))

    def add_depthmap(self, depth, image, scale, intrinsics, check_bound=False, seg=None, extrinsics=None):
        position = self.backproject(depth, scale, intrinsics)
        position = np.vstack((position, np.ones(len(position[0]))))
        position = np.dot(extrinsics, position)

        depth = np.divide(depth, scale)
        # print(depth.max(), depth.min())
        mask = (self.min_d <= depth) & (depth <= self.max_d)
        if self.roi is not None:
            mask[:self.roi[0], :] = False
            mask[self.roi[1]:, :] = False
            # mask[:, :self.roi[2]] = False
            # mask[:, self.roi[3]:] = False

        if seg is not None:
            mask_seg = np.zeros_like(mask)
            for id in seg_list:
                mask_seg |= (seg == id)

            mask = np.logical_and(mask_seg, mask)

        R = np.ravel(image[:, :, 0])
        G = np.ravel(image[:, :, 1])
        B = np.ravel(image[:, :, 2])

        points = np.transpose(np.vstack((position[0:3, :], R, G, B)))
        mask = mask.reshape((1, -1)).transpose(1, 0).repeat(6, axis = 1)
        # print(points.shape, mask.shape, points[mask].shape)
        # print(mask.reshape((depth.shape[0], -1)))
        points = points[mask]

        if not check_bound:
            self.data.extend(points.tolist())

        check_bound_points = points.reshape((-1, 6)).T[:3, :]
        # logger.debug("check_bound_points shape {}".format(check_bound_points.shape))
        # if check_bound_points.shape[1]:
        # 记录点云的最大最小边界
        self.voxel_bound[:, 0] = np.minimum(self.voxel_bound[:, 0], np.amin(check_bound_points, axis=1))
        self.voxel_bound[:, 1] = np.maximum(self.voxel_bound[:, 1], np.amax(check_bound_points, axis=1))

        return points.reshape((-1, 6))

    @staticmethod
    def draw_pointcloud(pcd_path, output = None):
        if Path(pcd_path).is_dir():
            root = Path(pcd_path)
            ply_path = root.glob("*.ply")

            vis = o3d.Visualizer()

            param = o3d.io.read_pinhole_camera_parameters('utils/viewpoint.json')
            vis.create_window(width=800, height=600)

            opt = vis.get_render_option()
            opt.point_size = 2
            # ctr一定要在create window后面
            ctr = vis.get_view_control()
            pointcloud = o3d.PointCloud()
            vis.add_geometry(pointcloud)

            to_reset = True

            for ply in sorted(ply_path):
                pcl = o3d.read_point_cloud(str(ply))
                # pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                vis.add_geometry(pcl)
                # 视角转换一定要加载add pcl的后面
                ctr.convert_from_pinhole_camera_parameters(param)
                vis.update_geometry()

                if output:
                    screen_path = Path(output) / ply.name
                    screen_path = screen_path.with_suffix(".jpg")
                    print(str(screen_path))
                    vis.capture_screen_image(str(screen_path), do_render=True)

                if to_reset:
                    vis.reset_view_point(True)
                    to_reset = False
                vis.poll_events()
                vis.update_renderer()
                vis.remove_geometry(pcl)
        else:
            pcl = o3d.read_point_cloud(str(pcd_path))
            pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d.draw_geometries([pcl])


    '''
    image_files: XXXXXX.png (RGB, 24-bit, PNG)
    depth_files: XXXXXX.png (16-bit, PNG)
    poses: camera-to-world, 4×4 matrix in homogeneous coordinates
    用来看单帧一个视角批量产生点云
    '''
    @staticmethod
    def build_single_frame_point_cloud(dataset_path, scale = 100.0):
        plyutils = PLYUtils(max_d=130)
        dataset_folder = Path(dataset_path)

        K = np.fromfile(os.path.join(dataset_path, "K.txt"), dtype=float, sep="\n ").reshape((3, 3))
        image_files = sorted((dataset_folder / "images").glob("*.png"))
        depth_files = sorted((dataset_folder / "depth_maps").glob("*.png"))
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
            save_ply_path = Path(dataset_path) / "point_clouds"
            # print(str(save_ply_path))
            # print(save_ply_path, save_ply_name)
            if not os.path.exists(str(save_ply_path)):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.mkdir(save_ply_path)
            PLYUtils.save(str(save_ply_path / save_ply_name), current_points_3D)

    @staticmethod
    def get_point_cloud_bound(image_files, depth_files, seg_files, K, poses, max_d=20, scale=100.0):
        plyutils = PLYUtils(max_d=max_d)

        # for i in tqdm(range(595,700)):
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
            plyutils.add_depthmap(depth, rgb, scale, check_bound=True, seg=seg, intrinsics=K, extrinsics=pose)
        print(plyutils.voxel_bound.T.flatten().tolist())
        return plyutils.voxel_bound.transpose().flatten().tolist()

class VOXELUtils(PLYUtils):
    def __init__(self, leaf_size, max_d, voxel_bound, cls=66):
        super(VOXELUtils, self).__init__()
        self.cls = cls
        self.max_d = max_d
        self.leaf_size = leaf_size
        self.voxel_bound = voxel_bound
        self.voxel_map = np.array([])
        self.x_min, self.y_min, self.z_min = self.voxel_bound[:3]
        self.x_max, self.y_max, self.z_max = self.voxel_bound[3:]

    def voxel_init(self):
        Dx = (self.x_max - self.x_min) // self.leaf_size + 5
        Dy = (self.y_max - self.y_min) // self.leaf_size + 5
        Dz = (self.z_max - self.z_min) // self.leaf_size + 5
        # 注意uint8可能会爆255
        self.voxel_map = np.zeros((int(Dx), int(Dy), int(Dz), self.cls), dtype=np.uint8)
        logger.info("voxel_map shape [{}, {}, {}]".format(Dx, Dy, Dz))

    def add_voxel(self, depth, seg, intrinsic, extrinsic, scale=100):
        position = self.backproject(depth, scale, intrinsic)
        position = np.vstack((position, np.ones(len(position[0]))))
        position = np.dot(extrinsic, position)

        logger.debug("pointcloud numpy array shape {}".format(position.shape))

        depth = np.divide(depth, scale)
        mask = (self.min_d <= depth) & (depth <= self.max_d)
        if self.roi is not None:
            mask[:self.roi[0], :] = False
            mask[self.roi[1]:, :] = False

        if seg is not None:
            mask_seg = np.zeros_like(mask)
            for id in seg_list:
                mask_seg |= (seg == id)

            mask = np.logical_and(mask_seg, mask)

        logger.debug("mask shape {}".format(mask.shape))
        seg = np.ravel(seg)

        points = np.transpose(np.vstack((position[0:3, :], seg)))

        logger.debug("point cloud with cls shape {}".format(points.shape))
        mask = mask.reshape((1, -1)).transpose(1, 0).repeat(4, axis=1)

        points = points[mask].reshape((-1, 4))
        # logger.debug("point cloud info {}".format(points.reshape((-1,4))[0]))

        point_index = (points[:, :3] - np.asarray([[self.x_min, self.y_min, self.z_min]]))
        point_index = np.floor_divide(point_index, self.leaf_size).astype(np.int)

        for idx in range(len(point_index)):
            x, y, z = point_index[idx]
            cls = int(points[idx, 3])
            # if x > 0 and y > 0 and z > 0:
            self.voxel_map[x, y, z, cls] += 1

    def voxel2bev(self, palette):
        virtical_sum = np.sum(self.voxel_map, axis=1)
        bev_map = np.argmax(virtical_sum, axis=2)

        img = np.zeros((bev_map.shape[0], bev_map.shape[1], 3), dtype=np.uint8)

        for label in show_seg_list:
            img[label == bev_map, :] = palette[label]
            # i += 1
        # for label, color in enumerate(palette):
        #     if label != 0:
        #         img[label == bev_map, :] = color
        # img = img[..., ::-1]

        return img


def main():
    K = np.fromfile("test_depth_dataset/kitti04/K.txt", dtype=float, sep="\n ").reshape((3, 3))
    image_files = sorted(Path("test_depth_dataset/kitti04/image_seg").glob("*.jpg"))
    depth_files = sorted(Path("test_depth_dataset/kitti04/depth_maps").glob("*.png"))
    seg_files = sorted(Path("test_depth_dataset/kitti04/segmentation").glob("*.png"))
    # 这里使用pose的真值
    poses = np.loadtxt("test_depth_dataset/kitti04/poses.txt").reshape((-1, 3, 4))

    # 先走一遍确定边界
    logger.info("waiting to initialize the voxel boundry")
    voxel_bound = PLYUtils.get_point_cloud_bound(image_files, depth_files, seg_files, K, poses, max_d=20)

    voxel = VOXELUtils(leaf_size=0.2, max_d=20, voxel_bound=voxel_bound)
    voxel.voxel_init()
    logger.info("voxel map shape : {}".format(voxel.voxel_map.shape))

    logger.info("waiting to generate the voxel map")
    # for idx in tqdm(range(595,700)):
    for idx in tqdm(range(len(poses))):
        image_file = str(image_files[idx])
        depth_file = str(depth_files[idx])
        seg_file = str(seg_files[idx])

        image = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)
        seg = cv2.imread(seg_file, 0)

        pose = np.concatenate((poses[idx], np.array([[0, 0, 0, 1]])), axis=0)
        height, weight, _ = image.shape
        depth = cv2.resize(depth, (weight, height))

        voxel.add_voxel(depth, seg, K, pose)
    img = voxel.voxel2bev(PALETTE)
    cv2.imwrite("bev.jpg", img)

    # args = parse_args()
    # PLYUtils.build_concat_cloud("test_depth_dataset/kitti04/", 'test04.ply')
    # PLYUtils.build_single_frame_point_cloud("test_depth_dataset/kitti09")
    # PLYUtils.draw_pointcloud('test_depth_dataset/kitti09/point_clouds', output="test_depth_dataset/kitti09/output")
    # PLYUtils.draw_pointcloud('test04_side.ply')
    # PLYUtils.draw_pointcloud('/data2/kitti/pointcloud/test12.ply')

if __name__ == '__main__':
    main()
