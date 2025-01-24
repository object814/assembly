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

            self.masked_pc_o3d_W = remove_outliers(self.masked_pc_o3d_W )

            self.rgb = rgb
            self.depth = depth
            self.K = self.multi_rs.camera_data[camera_idx]["pinhole_camera_intrinsic"]
            self.sv.scene.add_point_cloud(
                'object_pc',
                np.asarray(self.masked_pc_o3d_W.points),
                colors=np.asarray(self.masked_pc_o3d_W.colors),
                point_size=0.003,
                point_shape="circle"
            )

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

    
    def get_3d_point_from_2d(self, u, v,search_radius=5):
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
        # print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        height, width = depth.shape
        candidates = []  # 用于存储候选 3D 点

        for i in range(-search_radius, search_radius + 1):
            for j in range(-search_radius, search_radius + 1):
                # 计算邻近点的坐标
                u_new, v_new = int(u + i), int(v + j)

                # 检查边界条件
                if u_new < 0 or u_new >= width or v_new < 0 or v_new >= height:
                    continue

                # 获取深度值 Z
                Z = depth[v_new, u_new] / 1000.0  # 转换为米
                if Z == 0:
                    continue  # 忽略无效深度值

                # 计算 3D 坐标
                X = (u_new - cx) * Z / fx
                Y = (v_new - cy) * Z / fy
                point_3d_cam = [X,Y,Z]
                nX, nY, nZ = self.transform_to_world(point_3d_cam)
                candidates.append((nX, nY, nZ))

        # 检查候选点是否在物体点云范围内
        if not candidates:
            raise ValueError(f"No valid depth values around pixel ({u}, {v}) within radius {search_radius}")

        # 将物体点云范围提取为 AABB (轴对齐边界框)
        pc_min_bound = np.array(self.masked_pc_o3d_W.get_min_bound())
        pc_max_bound = np.array(self.masked_pc_o3d_W.get_max_bound())

        # 筛选在点云范围内的点
        valid_points = [
            np.array([x, y, z]) for x, y, z in candidates
            if np.all(pc_min_bound <= np.array([x, y, z])) and np.all(np.array([x, y, z]) <= pc_max_bound)
        ]

        if valid_points:
            # 返回第一个符合条件的点
            best_point = valid_points[0]
            print(f"Found valid 3D point in object bounds: {best_point}")
            return best_point
        else:
            # 返回最近的点（不在边界范围内）
            print("No valid points in object bounds, returning nearest candidate.")
            best_point = candidates[0]
        return np.array(best_point)
    def visualize_2d_and_3d(self, u, v,save_path="output",point_radius=0.005):
        """
        可视化 RGB 图像中的 2D 点和点云中的对应 3D 点
        :param u: 像素列坐标 (x)
        :param v: 像素行坐标 (y)
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 获取 3D 点
        point_3d = self.get_3d_point_from_2d(u, v)
        # point_3d = self.transform_to_world(point_3d_cam)

        point_3d_file = os.path.join(save_path, "3d_point.txt")
        with open(point_3d_file, "w") as f:
            f.write(f"3D Point (World): {point_3d.tolist()}\n")
            # f.write(f"3D Point (Camera): {point_3d_cam.tolist()}\n")
        print(f"3D point saved to {point_3d_file}")

        # 在 RGB 图像上标记 2D 点
        img_with_point = self.rgb.copy()
        cv2.circle(img_with_point, (int(u), int(v)), radius=5, color=(0, 0, 255), thickness=-1)
        
        rgb_file = os.path.join(save_path, "rgb_with_2d_point.png")
        cv2.imwrite(rgb_file, cv2.cvtColor(img_with_point, cv2.COLOR_RGB2BGR))
        print(f"RGB image with 2D point saved to {rgb_file}")

        plt.imshow(cv2.cvtColor(img_with_point, cv2.COLOR_BGR2RGB))
        plt.title(f"2D Point: ({u}, {v})")
        plt.axis("off")
        plt.show()
    #   

        # 创建高亮点的球体表示
        highlighted_color = [0, 1, 0] 
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=point_radius)
        sphere.translate(point_3d)  # 将球体移动到目标点位置
        sphere.paint_uniform_color(highlighted_color)  # 设置球体为绿色

        # 合并点云和高亮球体
        o3d.visualization.draw_geometries(
            [self.masked_pc_o3d_W, sphere],
            window_name="3D Visualization",
            point_show_normal=False,
        )
        # 保存点云
        point_cloud_file = os.path.join(save_path, "colored_point_cloud.ply")
        o3d.io.write_point_cloud(point_cloud_file, self.masked_pc_o3d_W)
        print(f"Colored point cloud saved to {point_cloud_file}")
         #在点云中高亮 3D 点
        single_point_cloud = o3d.geometry.PointCloud()
        single_point_cloud.points = o3d.utility.Vector3dVector([point_3d])
        single_point_cloud.paint_uniform_color([0, 1, 0])  # Green 高亮点

        #合并点云
        combined_point_cloud = self.masked_pc_o3d_W + single_point_cloud

        
        # 可视化点云
        o3d.visualization.draw_geometries(
            [combined_point_cloud],
            window_name="3D Visualization",
            point_show_normal=False,
        )

        # 保存点云
        point_cloud_file = os.path.join(save_path, "combined_point_cloud.ply")
        o3d.io.write_point_cloud(point_cloud_file, combined_point_cloud)
        print(f"Combined point cloud saved to {point_cloud_file}")
            
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
    for idx in range(5):
        realtime_PC.get_now_pcd_from_camera()
    #     realtime_PC.vis_PC()
    #     realtime_PC.save_pc(save_dir="/home/shaol/data/zhoujx/Obj_mesh_dir/Box_02", file_name=f"Box_02.ply")

    # pcs=realtime_PC.load_pc(load_dir="/home/shaol/data/zhoujx/Obj_mesh_dir/Box_02", file_name=f"Box_02.ply")
    # realtime_PC.vis_PC()
    point1=[400, 570]
    point2=[600, 445]
    
    u, v = point1
    try:
        # 可视化 2D 和 3D 点
        realtime_PC.visualize_2d_and_3d(u,v,save_path='/home/shaol/data/zhoujx/Obj_mesh_dir/Box_02')
    except ValueError as e:
        print(e)