import sys

sys.path.append(".")
import os

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
        self.prompt_drawer = SAM2PromptDrawer(window_name="Prompt Drawer",
                                              screen_scale=2.0,
                                              sam_checkpoint=SAM_PATH,
                                              device="cuda",
                                              model_type=SAM_TYPE)

        arm_right_cam_serial = "241122074374"  # 241122074374, 233622079809
        # top_cam_serial = '233622079809'
        camera_serial_nums = [arm_right_cam_serial]
        # camera_serial_nums = [top_cam_serial]
        self.multi_rs = MultiRealsense(camera_serial_nums)
        # arm_right_cam_K_path = Path("3rdparty/xarm6/data/camera/mounted_white/K.npy")
        arm_right_cam_K_path = Path(f"3rdparty/xarm6/data/camera/{arm_right_cam_serial}/K.npy")
        # arm_right_cam_K_path = Path("3rdparty/xarm6/data/camera/mounted_top/K.npy")
        arm_right_cam_K = np.load(arm_right_cam_K_path)
        # arm_right_cam_X_BaseCamera_path = Path("3rdparty/xarm6/data/camera/mounted_white/0922_excalib_capture00/optimized_X_BaseCamera.npy")
        arm_right_cam_X_BaseCamera_path = Path(
            f"3rdparty/xarm6/data/camera/{arm_right_cam_serial}/1113_excalib_capture02/optimized_X_BaseCamera.npy")
        # arm_right_cam_X_BaseCamera_path = Path("3rdparty/xarm6/data/camera/mounted_top/1109_excalib_capture00/optimized_X_BaseCamera.npy")
        arm_right_cam_X_BaseCamera = np.load(arm_right_cam_X_BaseCamera_path)
        self.multi_rs.set_intrinsics(0, arm_right_cam_K[0, 0], arm_right_cam_K[1, 1], arm_right_cam_K[0, 2],
                                     arm_right_cam_K[1, 2])
        self.camera_wxyzs = [
            R.from_matrix(arm_right_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
        ]
        self.camera_positions = [arm_right_cam_X_BaseCamera[:3, 3]]
        self.X_BaseCamera_list = [arm_right_cam_X_BaseCamera]

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
                                wxyz=R.from_matrix(arm_right_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                position=arm_right_cam_X_BaseCamera[:3, 3])
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
            if self.if_init:
                mask_np = self.prompt_drawer.run(rgb)  # (720, 1280)
                self.if_init = False
            else:
                mask_np = self.prompt_drawer.track(rgb)
            # mask_np = shrink_mask(mask_np, shrink_coefficient=0.9)
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


if __name__ == "__main__":
    realtime_PC = Realtime_PC()
    id = 0
    while True:
        realtime_PC.get_now_pcd_from_camera()
        realtime_PC.vis_PC()
        # realtime_PC.save_pc(save_dir="data/saved_pc/blue _short", file_name=f"blue _short_{id}.ply")

        # ### load the point cloud from the saved file
        # # realtime_PC.load_pc(load_dir="data/saved_pc", file_name="long_pants_v.ply")
        # realtime_PC.load_pc(load_dir="data/saved_pc/long_pants_1", file_name=f"long_pant_{id}.ply")
        # realtime_PC.vis_PC()
        # if id == 0:
        #     time.sleep(5)
        # time.sleep(1)
        # id += 1

    # prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer", screen_scale=2.0, sam_checkpoint=SAM_PATH, device="cuda", model_type=SAM_TYPE)
    prompt_drawer = SAM2PromptDrawer(window_name="Prompt Drawer",
                                     screen_scale=2.0,
                                     sam_checkpoint=SAM_PATH,
                                     device="cuda",
                                     model_type=SAM_TYPE)
    # prompt_drawer = SAMAutoDrawer(window_name="Prompt Drawer", screen_scale=2.0, sam_checkpoint=SAM_PATH, device="cuda", model_type=SAM_TYPE)

    arm_right_cam_serial = "241122074374"
    camera_serial_nums = [arm_right_cam_serial]
    multi_rs = MultiRealsense(camera_serial_nums)
    arm_right_cam_K_path = Path("3rdparty/xarm6/data/camera/mounted_white/K.npy")
    arm_right_cam_K = np.load(arm_right_cam_K_path)
    arm_right_cam_X_BaseCamera_path = Path(
        "3rdparty/xarm6/data/camera/mounted_white/0922_excalib_capture00/optimized_X_BaseCamera.npy")
    arm_right_cam_X_BaseCamera = np.load(arm_right_cam_X_BaseCamera_path)
    multi_rs.set_intrinsics(0, arm_right_cam_K[0, 0], arm_right_cam_K[1, 1], arm_right_cam_K[0, 2], arm_right_cam_K[1,
                                                                                                                    2])
    camera_wxyzs = [
        R.from_matrix(arm_right_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
    ]
    camera_positions = [arm_right_cam_X_BaseCamera[:3, 3]]
    X_BaseCamera_list = [arm_right_cam_X_BaseCamera]

    sv = viser.ViserServer(port=8080)

    for _ in range(50):
        multi_rs.getCurrentData()
    if_init = True

    while True:
        rtr_dict_list = multi_rs.getCurrentData()
        for camera_idx in range(len(rtr_dict_list)):
            rtr_dict = rtr_dict_list[camera_idx]

            rgb = rtr_dict["rgb"]
            depth = (rtr_dict["depth"].astype(np.float32)).astype(np.float32)
            # pc_o3d = rtr_dict["pointcloud_o3d"]

            prompt_drawer.reset()
            if if_init:
                mask_np = prompt_drawer.run(rgb)  # (720, 1280)
                if_init = False
            else:
                mask_np = prompt_drawer.track(rgb)

            h, w = mask_np.shape[-2:]

            masked_pc_o3d = get_masked_pointcloud(rgb, depth, mask_np.reshape(h, w),
                                                  multi_rs.camera_data[camera_idx]["pinhole_camera_intrinsic"])
            X_BaseCamera = X_BaseCamera_list[camera_idx]
            rs_pc_in_C_np = np.asarray(masked_pc_o3d.points)
            rs_pc_in_B = X_BaseCamera @ np.vstack([rs_pc_in_C_np.T, np.ones(rs_pc_in_C_np.shape[0])])
            rs_pc_in_B = rs_pc_in_B[:3].T
            masked_pc_o3d_W = o3d.geometry.PointCloud()
            masked_pc_o3d_W.points = o3d.utility.Vector3dVector(rs_pc_in_B)
            masked_pc_o3d_W.colors = masked_pc_o3d.colors

        sv.scene.add_point_cloud('object_pc',
                                 np.asarray(masked_pc_o3d_W.points),
                                 colors=np.asarray(masked_pc_o3d_W.colors),
                                 point_size=0.003,
                                 point_shape="circle")
