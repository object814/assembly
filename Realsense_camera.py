import sys

sys.path.append(".")
sys.path.append("/home/shaol/data/zhoujx/rw/third_party/xarm6")

import os

import matplotlib.pyplot as plt
import cv2
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
from xarm6_interface.utils.sam_prompt_drawer import SAMPromptDrawer, SAM2PromptDrawer, shrink_mask
from xarm6_interface.utils.realsense import MultiRealsense, get_masked_pointcloud, remove_outliers
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from xarm6_interface import SAM_TYPE, SAM_PATH
import open3d as o3d
import viser
import sys
import time

from xarm6_interface.arm_pk import XArm6WOEE
from xarm6_interface.utils import as_mesh
import math

# 角度转弧度
angle_in_degrees = 45
angle_in_radians = math.radians(angle_in_degrees)


class Realtime_PC:

    def __init__(self):

        # self.prompt_drawer = SAM2PromptDrawer(window_name="Prompt Drawer",
        #                                       screen_scale=2.0,
        #                                       sam_checkpoint=SAM_PATH,
        #                                       device="cuda",
        #                                       model_type=SAM_TYPE)

        self.prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer",
                                    screen_scale=2.0,
                                    sam_checkpoint=SAM_PATH,
                                    device="cuda",
                                    model_type=SAM_TYPE)

        front_cam_serial = "317222074181"  # 241122074374, 233622079809
        exp_name = "0124_excalib_capture00"
        left_exp_name = "0124_excalib_capture00_Left"
        # camera_serial_nums = [arm_right_cam_serial]
        camera_serial_nums = [front_cam_serial]
        # camera_serial_nums = [top_cam_serial]
        self.multi_rs = MultiRealsense(camera_serial_nums)

        # arm_right_cam_K_path = Path(f"3rdparty/xarm6/data/camera/{arm_right_cam_serial}/K.npy")
        front_cam_K_path = Path(f"third_party/xarm6/data/camera/{front_cam_serial}/K.npy")

        # arm_right_cam_K = np.load(arm_right_cam_K_path)
        front_cam_K = np.load(front_cam_K_path)
        
        
        front_cam_X_BaseCamera_path = Path(f"third_party/xarm6/data/camera/{front_cam_serial}/{exp_name}/optimized_X_BaseCamera.npy")
        Left_front_cam_X_BaseCamera_path = Path(f"third_party/xarm6/data/camera/{front_cam_serial}/{left_exp_name}/optimized_X_BaseCamera.npy")
        
        front_cam_X_BaseCamera = np.load(front_cam_X_BaseCamera_path)

        self.multi_rs.set_intrinsics(0, front_cam_K[0, 0], front_cam_K[1, 1], front_cam_K[0, 2],
                                     front_cam_K[1, 2])
        self.camera_wxyzs = [
            R.from_matrix(front_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
        ]
        self.camera_positions = [front_cam_X_BaseCamera[:3, 3]]
        self.X_BaseCamera_list = [front_cam_X_BaseCamera]
        self.Extrinsic_matrix = front_cam_X_BaseCamera
        self.Intrinsic_matrix = front_cam_K
        self.camera_serial = front_cam_serial
        self.camera_name = "front_cam"
        self.camera_id = 0
        self.camera_type = "realsense"
        self.camera_wxyz = R.from_matrix(front_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self.X_BaserightBaseleft = np.array([[-0.9994789, -0.02021456, -0.02516249, 1.1056788],
                                             [0.02050998, -0.99972314, -0.01153819, -0.03005717],
                                             [-0.02492228, -0.01204827, 0.9996168, 0.00512242], [0.0, 0.0, 0.0, 1.0]])

        self.sv = viser.ViserServer(port=8080)

        rw_joint_values_in_degree = [0, -53.7, -47.1, 0, 100.9, 0]
        rw_joint_values = [math.radians(angle) for angle in rw_joint_values_in_degree]
        arm_right = XArm6WOEE()
        rw_arm_right_visual_mesh = as_mesh(arm_right.get_state_trimesh(rw_joint_values)['visual'])
        arm_left = XArm6WOEE()
        rw_arm_left_visual_mesh = as_mesh(arm_left.get_state_trimesh(rw_joint_values)['visual'])

        self.sv.scene.add_mesh_simple("rw_arm_right",
                                      vertices=rw_arm_right_visual_mesh.vertices,
                                      faces=rw_arm_right_visual_mesh.faces,
                                      color=[0.7, 0.7, 0.7],
                                      opacity=0.5,
                                      wxyz=(1, 0, 0, 0),
                                      position=(0, 0, 0))

        self.sv.scene.add_mesh_simple("rw_arm_left",
                                      vertices=rw_arm_left_visual_mesh.vertices,
                                      faces=rw_arm_left_visual_mesh.faces,
                                      color=[0.7, 0.7, 0.7],
                                      opacity=0.5,
                                      wxyz=R.from_matrix(self.X_BaserightBaseleft[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                      position=(self.X_BaserightBaseleft[:3, 3]))

        self.sv.scene.add_frame("camera",
                                wxyz=R.from_matrix(front_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                position=front_cam_X_BaseCamera[:3, 3])
        self.sv.scene.add_frame("left",
                                wxyz=R.from_matrix(self.X_BaserightBaseleft[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                position=self.X_BaserightBaseleft[:3, 3])
        # while True:
        #     time.sleep(0.1)

        for _ in range(50):
            self.multi_rs.getCurrentData()
        self.if_init = True

        # self.get_now_pcd_from_camera()

    def get_now_pcd_from_camera(self):
        rtr_dict_list = self.multi_rs.getCurrentData()
        for camera_idx in range(len(rtr_dict_list)):
            rtr_dict = rtr_dict_list[camera_idx]
            rgb = rtr_dict["rgb"]
            depth = (rtr_dict["depth"].astype(np.float32)).astype(np.float32)
            # pc_o3d = rtr_dict["pointcloud_o3d"]

            self.prompt_drawer.reset()
            mask_np = self.prompt_drawer.run(rgb) 
            # if self.if_init:
            #     mask_np = self.prompt_drawer.run(rgb)  # (720, 1280)
            #     self.if_init = False
            # else:
            #     mask_np = self.prompt_drawer.track(rgb)
            mask_np = shrink_mask(mask_np, shrink_coefficient=0.9)
            h, w = mask_np.shape[-2:]

            masked_pc_o3d = get_masked_pointcloud(rgb, depth, mask_np.reshape(h, w),
                                                  self.multi_rs.camera_data[camera_idx]["pinhole_camera_intrinsic"])
            X_BaseCamera = self.X_BaseCamera_list[camera_idx]
            rs_pc_in_C_np = np.asarray(masked_pc_o3d.points)
            rs_pc_in_B = X_BaseCamera @ np.vstack([rs_pc_in_C_np.T, np.ones(rs_pc_in_C_np.shape[0])])
            rs_pc_in_B = rs_pc_in_B[:3].T
            self.masked_pc_o3d_W = o3d.geometry.PointCloud()
            self.masked_pc_o3d_W.points = o3d.utility.Vector3dVector(rs_pc_in_B)
            self.masked_pc_o3d_W.colors = masked_pc_o3d.colors

            self.rgb = rgb
            self.depth = depth
            self.K = self.multi_rs.camera_data[camera_idx]["pinhole_camera_intrinsic"]
    def get_masked_pc(self):

        self.vis_PC()
        return self.masked_pc_o3d_W

    def get_sv(self):
        return self.sv

    def vis_PC(self):
        self.sv.scene.add_point_cloud('object_pc',
                                      np.asarray(self.masked_pc_o3d_W.points),
                                      colors=np.asarray(self.masked_pc_o3d_W.colors),
                                      point_size=0.003,
                                      point_shape="circle")

    def save_pc(self, save_dir, file_name):
        # save_dir = "data/saved_pc"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        o3d.io.write_point_cloud(save_path, self.masked_pc_o3d_W)

    def load_pc(self, load_dir, file_name):
        load_path = os.path.join(load_dir, file_name)
        self.masked_pc_o3d_W = o3d.io.read_point_cloud(load_path)
        return self.masked_pc_o3d_W 

    def get_3d_points_on_object(self, pixel_coords, object_mask, camera_idx=0):
        """
        从 2D 图像像素坐标匹配到物体点云上的 3D 点。

        Args:
            pixel_coords (np.ndarray): 输入的 2D 图像像素坐标，形状为 (N, 2)，每行 [x, y]。
            object_mask (np.ndarray): 物体掩码，形状与图像一致，物体区域为 True，其余为 False。
            camera_idx (int): 指定使用第几个相机的数据，默认为第一个相机。

        Returns:
            np.ndarray: 对应的 3D 点，形状为 (N, 3)，相机坐标系下的物体点云。
        """
        # 获取当前摄像头数据
        rtr_dict_list = self.multi_rs.getCurrentData()
        rtr_dict = rtr_dict_list[camera_idx]
        rgb = rtr_dict["rgb"]
        depth = rtr_dict["depth"].astype(np.float32)

        # 相机内参
        intrinsic = self.multi_rs.camera_data[camera_idx]["pinhole_camera_intrinsic"]
        fx, fy = intrinsic.intrinsic_matrix[0, 0], intrinsic.intrinsic_matrix[1, 1]
        cx, cy = intrinsic.intrinsic_matrix[0, 2], intrinsic.intrinsic_matrix[1, 2]

        # 筛选出物体点云的深度图
        depth[~object_mask] = 0  # 将非物体区域的深度值置为 0

        # 投影 2D 坐标到 3D 点云
        points_3d_camera = []
        for x, y in pixel_coords:
            z = depth[int(y), int(x)]  # 获取深度值
            if z == 0:  # 跳过无效点
                continue
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            points_3d_camera.append([x_3d, y_3d, z])

        points_3d_camera = np.array(points_3d_camera)
        if points_3d_camera.size == 0:
            print("未找到有效的 3D 点，请检查输入的 2D 坐标或物体掩码。")
            return np.array([])

        return points_3d_camera

    def visualize_object_points(self, pixel_coords, object_mask, camera_idx=0):
        """
        可视化从 2D 图像匹配到的物体点云上的 3D 点。

        Args:
            pixel_coords (np.ndarray): 输入的 2D 图像像素坐标，形状为 (N, 2)。
            object_mask (np.ndarray): 物体掩码，形状与图像一致。
            camera_idx (int): 指定使用第几个相机的数据，默认为第一个相机。
        """
        # 获取物体点云的 3D 点
        points_3d = self.get_3d_points_on_object(pixel_coords, object_mask, camera_idx)
        if points_3d.size == 0:
            print("没有找到匹配的 3D 点。")
            return

        # 获取完整物体点云
        rtr_dict_list = self.multi_rs.getCurrentData()
        rtr_dict = rtr_dict_list[camera_idx]
        depth = rtr_dict["depth"].astype(np.float32)
        depth[~object_mask] = 0  # 掩码区域保留，非物体区域置零
        full_object_pointcloud = get_masked_pointcloud(
            rtr_dict["rgb"], depth, object_mask, self.multi_rs.camera_data[camera_idx]["pinhole_camera_intrinsic"])

        # 创建 Open3D 点云对象
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(np.asarray(full_object_pointcloud.points))
        object_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

        highlight_pcd = o3d.geometry.PointCloud()
        highlight_pcd.points = o3d.utility.Vector3dVector(points_3d)
        highlight_pcd.paint_uniform_color([1, 0, 0])  # 红色高亮点

        # 可视化
        o3d.visualization.draw_geometries([object_pcd, highlight_pcd], window_name="2D -> 3D 点匹配可视化（物体点云）")
    
    def get_3d_point_from_2d(self, u, v):
        """
        根据 2D 像素坐标 (u, v)，获取对应点云中的 3D 坐标。
        :param u: 像素列坐标 (x)
        :param v: 像素行坐标 (y)
        :return: 对应的 3D 点 (X, Y, Z)
        """
        # 提取深度图和内参矩阵
        depth = self.multi_rs.camera_data[0]["depth"]
        K = self.multi_rs.camera_data[0]["pinhole_camera_intrinsic"].intrinsic_matrix

        # 获取相机内参
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        # 深度值 Z
        Z = depth[int(v), int(u)]/ 1000.0
        if Z == 0:
            raise ValueError(f"No valid depth value at pixel ({u}, {v})")

        # 计算 3D 坐标
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        # 打印 3D 点和点云范围
        print(f"Generated 3D point: [{X}, {Y}, {Z}]")
        print("Point cloud bounds:")
        print(self.masked_pc_o3d_W.get_min_bound(), self.masked_pc_o3d_W.get_max_bound())
        return np.array([X, Y, Z])
    def visualize_2d_and_3d(self, u, v):
        """
        可视化 RGB 图像中的 2D 点和点云中的对应 3D 点
        :param u: 像素列坐标 (x)
        :param v: 像素行坐标 (y)
        """
        # 获取 3D 点
        point_3d_cam = self.get_3d_point_from_2d(u, v)
        point_3d = self.transform_to_world(point_3d_cam)
        # 在 RGB 图像上标记 2D 点
        img_with_point = self.rgb.copy()
        cv2.circle(img_with_point, (int(u), int(v)), radius=5, color=(0, 0, 255), thickness=-1)
        plt.imshow(cv2.cvtColor(img_with_point, cv2.COLOR_BGR2RGB))
        plt.title(f"2D Point: ({u}, {v})")
        plt.axis("off")
        plt.show()

        # 在点云中高亮 3D 点
        single_point_cloud = o3d.geometry.PointCloud()
        single_point_cloud.points = o3d.utility.Vector3dVector([point_3d])
        single_point_cloud.paint_uniform_color([0, 1, 0])  # green高亮点

        # 合并点云
        combined_point_cloud = self.masked_pc_o3d_W + single_point_cloud
        o3d.visualization.draw_geometries(
            [combined_point_cloud],
            window_name="3D Visualization",
            point_show_normal=False,
        )
    def transform_to_world(self, point_3d):
        """
        将 3D 点从相机坐标系转换到点云的世界坐标系。
        :param point_3d: 相机坐标系下的 3D 点 (X, Y, Z)
        :return: 世界坐标系下的 3D 点
        """
        # 获取外参矩阵
        extrinsic = self.Extrinsic_matrix
        # 转换为齐次坐标
        point_camera = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])  # 添加齐次维度

        # 应用外参变换
        point_world = extrinsic @ point_camera

        return point_world[:3]  
if __name__ == "__main__":
    realtime_PC = Realtime_PC()
    id = 0
    # realtime_PC.multi_rs.create_window_and_capture_data(save_dir=Path("/home/shaol/data/zhoujx/Obj_mesh_dir/Box_02"))
    # while True:
    realtime_PC.get_now_pcd_from_camera()
    #     realtime_PC.vis_PC()
    #     realtime_PC.save_pc(save_dir="/home/shaol/data/zhoujx/Obj_mesh_dir/Box_02", file_name=f"Box_02.ply")

    ### load the point cloud from the saved file
    realtime_PC.load_pc(load_dir="data/saved_pc", file_name="long_pants_v.ply")
    pcs=realtime_PC.load_pc(load_dir="/home/shaol/data/zhoujx/Obj_mesh_dir/Box_02", file_name=f"Box_02.ply")
    # realtime_PC.vis_PC()
    u, v = 600, 600

    try:
        # 可视化 2D 和 3D 点
        realtime_PC.visualize_2d_and_3d(u, v)
    except ValueError as e:
        print(e)